from __future__ import annotations
import argparse
import json
import logging
import random
import time
from pathlib import Path
import numpy as np
import torch
from delivery_environment import PureRealRoadNetworkEnvironmentWithConstraints
from distance_store import distance_store
from slotmatch_policy import StructuredMatchingHRL
from historical_prototypes import build_train_historical_prototypes
from imitation_training import collect_optimizer_demonstrations, pretrain_from_optimizer
log = logging.getLogger('smhrl')

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Experiment:

    def __init__(self, total_vehicles: int, seed: int, epochs: int, output: Path):
        assert total_vehicles % 2 == 0
        self.total = total_vehicles
        self.n_uav = total_vehicles // 2
        self.n_carrier = total_vehicles // 2
        self.seed = seed
        self.epochs = epochs
        self.output = output / f'nums_{total_vehicles}'
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = PureRealRoadNetworkEnvironmentWithConstraints(self.n_carrier, self.n_uav, 50, 15)
        state = self.env.reset(day=1)
        prototypes = build_train_historical_prototypes(Path('shanghai_road_dataset'))
        self.agent = StructuredMatchingHRL(len(state), self.n_uav, self.n_carrier, max_orders=15, historical_prototypes=prototypes, device='cuda' if torch.cuda.is_available() else 'cpu')

    def train_epoch(self, epoch: int):
        day = random.choice(list(range(1, 22)))
        state = self.env.reset(day=day)
        reward_sum = 0.0
        previous_pn = 0
        previous_dn = 0
        started = time.perf_counter()
        info = {}
        for step in range(50):
            action = self.agent.get_action(state, self.env, mode='train')
            (next_state, reward, done, info) = self.env.step(action)
            current_pn = int(self.env.episode_pickups)
            current_dn = int(self.env.episode_deliveries)
            delta_pn = current_pn - previous_pn
            delta_dn = current_dn - previous_dn
            learning_reward = 1.5 * float(info.get('new_orders_count', 0) > 0) + 10.0 * delta_pn + 10.0 * delta_dn - 0.12 * float(info.get('active_orders', 0)) - 0.08 * len(self.env.vehicle_assigned_orders)
            self.agent.store_transition(learning_reward, done)
            (previous_pn, previous_dn) = (current_pn, current_dn)
            reward_sum += reward
            state = next_state
            if done:
                break
        losses = self.agent.update()
        return {'epoch': epoch, 'day': day, 'reward': reward_sum, 'pn': self.env.episode_pickups, 'dn': self.env.episode_deliveries, 'orders': self.env.total_orders_generated, 'seconds': time.perf_counter() - started, 'losses': losses}
