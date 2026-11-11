from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F

from assignment_teacher import oracle_action
from delivery_environment import VehicleState


def collect_optimizer_demonstrations(agent, env, days=range(1, 22), seed=10):
    """Collect state -> globally optimized edge labels on training days only."""
    examples = []
    for day in days:
        random.seed(seed + day); np.random.seed(seed + day); torch.manual_seed(seed + day)
        state = env.reset(day=day)
        history = np.asarray(state, np.float32).copy()
        for _ in range(env.max_time_steps):
            state_np = np.asarray(state, np.float32)
            if env.time_step:
                history = 0.85 * history + 0.15 * state_np
            policy_input = np.concatenate([state_np, history]).astype(np.float32)
            action = oracle_action(env, uav_only=True)
            dispatch, uw, cw = action
            uavs = sorted([v for v in env.vehicles if v.vehicle_type == "uav" and
                           env.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE], key=lambda v: v.vehicle_id)
            carriers = sorted([v for v in env.vehicles if v.vehicle_type == "carrier" and
                               env.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE], key=lambda v: v.vehicle_id)
            um = np.asarray(uw).reshape(env.num_uavs, 15)
            cm = np.asarray(cw).reshape(env.num_ground_vehicles, 15)
            choices = np.full(15, -1, np.int64)
            modes = np.zeros(15, np.int64)
            for slot in range(15):
                ui = np.flatnonzero(um[:len(uavs), slot] > 0.5)
                ci = np.flatnonzero(cm[:len(carriers), slot] > 0.5)
                if len(ui):
                    choices[slot] = uavs[int(ui[0])].vehicle_id - env.num_ground_vehicles
                    modes[slot] = 0
                elif len(ci):
                    choices[slot] = env.num_uavs + carriers[int(ci[0])].vehicle_id
                    modes[slot] = 1
            examples.append((policy_input, choices, modes))
            state, _, done, _ = env.step(action)
            if done: break
    return examples


def pretrain_from_optimizer(agent, examples, epochs=3, batch_size=64):
    states = torch.as_tensor(np.stack([x[0] for x in examples]), device=agent.device)
    choices = torch.as_tensor(np.stack([x[1] for x in examples]), device=agent.device)
    modes = torch.as_tensor(np.stack([x[2] for x in examples]), device=agent.device)
    last = {}
    for epoch in range(epochs):
        order = torch.randperm(len(states), device=agent.device)
        slotmatchning = []
        for start in range(0, len(states), batch_size):
            idx = order[start:start+batch_size]
            mode_logits, edge_logits, _, _, _ = agent.policy(states[idx])
            valid = choices[idx] >= 0
            if not valid.any(): continue
            # [B,V,S] -> one categorical vehicle distribution per slot.
            edge_by_slot = edge_logits.permute(0, 2, 1)
            edge_loss = F.cross_entropy(edge_by_slot[valid], choices[idx][valid])
            mode_loss = F.cross_entropy(mode_logits[valid], modes[idx][valid])
            loss = edge_loss + 0.5 * mode_loss
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            agent.optimizer.step()
            slotmatchning.append(float(loss.item()))
        last = {"epoch": epoch, "loss": float(np.mean(slotmatchning))}
    return last
