from __future__ import annotations
import argparse, hashlib, json, logging, os, random, sys, time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train','evaluate'])
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'checkpoints/slotmatch_15uav_15ground_seed1.pt')
    parser.add_argument('--output',type=Path,default=Path('slotmatchs'))
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--fleet',type=int,default=30)
    parser.add_argument('--epochs',type=int,choices=range(1,11),default=10)
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    args.checkpoint=args.checkpoint.resolve();data=args.data_root.resolve()
    if data.name!='shanghai_road_dataset':raise ValueError('--data-root must point to shanghai_road_dataset')
    needed=['road_network/road_graph_fixed.pkl','road_network/node_features_fixed.npy','road_network/nodes_processed.csv','mapping/grid_to_node_mapping.csv']
    days=range(1,31) if args.mode=='train' else list(range(1,22))+[28,29,30]
    needed += [f'processed_orders_road/Orders_Dataset_shanghai_road_day_{day}' for day in days]
    missing=[p for p in needed if not (data/p).is_file()]
    if missing:raise FileNotFoundError(str(missing))
    os.chdir(data.parent)
    global Experiment,seed_all,collect_optimizer_demonstrations,pretrain_from_optimizer,distance_store
    from slotmatch_trainer import Experiment,seed_all
    from imitation_training import collect_optimizer_demonstrations,pretrain_from_optimizer
    from distance_store import distance_store
    import spatial_metrics
    from shortest_path_memo import memoize_dijkstra
    if spatial_metrics.scipy_dijkstra is not None:spatial_metrics.scipy_dijkstra=memoize_dijkstra(spatial_metrics.scipy_dijkstra)
    distance_store.cache_file=str(args.output/'distance_store.pkl')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    logging.getLogger().setLevel(logging.ERROR)
    if args.mode=='train':slotmatch_one('SMHRL',args.fleet,args.seed,args)
    else:
        seed_all(args.seed);trainer=Experiment(args.fleet,args.seed,10,args.output/'scratch')
        trainer.agent.load(args.checkpoint)
        result=evaluate(trainer.agent,trainer.env,'SMHRL',TEST,args.seed)
        write_json(args.output/'evaluation.json',result)
        print(json.dumps(result,indent=2))

TRAIN=tuple(range(1,22));VAL=tuple(range(22,28));TEST=(28,29,30)
@contextmanager
def preserve_rng():
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def evaluate(agent, env, method, days, seed):
    rows = []
    modules = [agent.policy]
    modes = [m.training for m in modules]
    with preserve_rng():
        try:
            for module in modules:
                module.eval()
            for day in days:
                seed_all(seed * 1000 + day)
                state = env.reset(day=day)
                assert env.loaded_day == day
                initial_nodes = [int(v.current_node) for v in env.vehicles]
                action_seconds = env_seconds = 0.0
                for step in range(50):
                    sync()
                    start = time.perf_counter()
                    action = agent.get_action(state, env, mode='test')
                    sync()
                    action_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    (state, _, done, _) = env.step(action)
                    env_seconds += time.perf_counter() - start
                    if done:
                        break
                assert env.episode_deliveries == env.episode_uav_deliveries + env.episode_carrier_deliveries
                rows.append(dict(day=day, pn=int(env.episode_pickups), dn=int(env.episode_deliveries), orders=int(env.total_orders_generated), steps=step + 1, initial_nodes=initial_nodes, action_seconds_per_step=action_seconds / (step + 1), environment_seconds_per_step=env_seconds / (step + 1)))
        finally:
            for (module, mode) in zip(modules, modes):
                module.train(mode)
            agent._pending = None
    return dict(days=rows, mean_pn=float(np.mean([r['pn'] for r in rows])), mean_dn=float(np.mean([r['dn'] for r in rows])))

def selection_key(row):
    return (row['mean_pn'] + row['mean_dn'], row['mean_dn'], row['mean_pn'])

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def slotmatch_one(method, fleet, seed, args):
    output = args.output / method / f'seed_{seed}' / f'nums_{fleet}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed slotmatch: {output}')
    seed_all(seed)
    started = time.perf_counter()
    trainer = Experiment(fleet, seed, args.epochs, output / 'scratch')
    (agent, env) = (trainer.agent, trainer.env)
    print(f'START {method} fleet={fleet} seed={seed}', flush=True)
    pretraining = None
    prestart = time.perf_counter()
    examples = collect_optimizer_demonstrations(agent, env, days=TRAIN, seed=seed)
    stats = pretrain_from_optimizer(agent, examples, epochs=3)
    pretraining = dict(epochs=3, demonstration_days=list(TRAIN), demonstration_transitions=len(examples), seconds=time.perf_counter() - prestart, final_stats=stats)
    print(f"PRETRAIN DONE fleet={fleet} seed={seed} seconds={pretraining['seconds']:.1f}", flush=True)
    (history, best) = ([], None)
    checkpoint = output / 'best_val.pt'
    training_rng = random.Random(seed)
    training_seeds = [training_rng.randrange(1, 2 ** 31 - 1) for _ in range(args.epochs)]
    for epoch in range(args.epochs):
        seed_all(training_seeds[epoch])
        tick = time.perf_counter()
        agent.policy.train()
        train = trainer.train_epoch(epoch)
        val = evaluate(agent, env, method, VAL, seed)
        row = dict(epoch=epoch + 1, train=train, validation=val, training_seed=training_seeds[epoch], seconds=time.perf_counter() - tick)
        history.append(row)
        if best is None or selection_key(val) > selection_key(best['validation']):
            best = row
            agent.save(checkpoint)
        write_json(output / 'validation_history.json', history)
        print(f"VAL {method} fleet={fleet} seed={seed} epoch={epoch + 1} PN={val['mean_pn']:.2f} DN={val['mean_dn']:.2f} best={best['epoch']} seconds={row['seconds']:.1f}", flush=True)
    agent.load(checkpoint)
    test = evaluate(agent, env, method, TEST, seed)
    distance_store.save_cache()
    result = dict(method=method, fleet=fleet, seed=seed, protocol='validation-selected; frozen model; historical test-guided development disclosed', train_days=list(TRAIN), val_days=list(VAL), test_days=list(TEST), rl_epochs=args.epochs, pretraining=pretraining, selection_rule='max validation mean(PN+DN), then DN, PN, earliest epoch', selected_epoch=best['epoch'], selected_validation=best['validation'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), test=test, history=history, seconds=time.perf_counter() - started, slotmatchtime=dict(torch=torch.__version__, cuda=torch.version.cuda, exact_dijkstra_memoization=True, device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'))
    write_json(output / 'metrics.json', result)
    print(f"DONE {method} fleet={fleet} seed={seed} selected={best['epoch']} TEST PN={test['mean_pn']:.2f} DN={test['mean_dn']:.2f}", flush=True)

if __name__=='__main__':main()
