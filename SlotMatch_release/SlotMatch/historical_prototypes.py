from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np

from spatial_metrics import distance_calculator


def build_train_historical_prototypes(dataset_root: Path, days=range(1, 22), horizon: int = 50):
    """Return per-time, per-arrival-slot medoids using training days only."""
    samples = {(scenario, t): {} for scenario in range(6) for t in range(horizon)}
    base = dataset_root / "processed_orders_road"
    for day in days:
        with (base / f"Orders_Dataset_shanghai_road_day_{day}").open("rb") as handle:
            day_data = pickle.load(handle)
        for t in range(horizon):
            records = []
            for payload in day_data.get(t, {}).values():
                if isinstance(payload, list) and payload:
                    row = payload[0] if isinstance(payload[0], list) else payload
                    if len(row) >= 5:
                        records.append(row)
            records.sort(key=lambda row: int(row[0]))
            for slot, row in enumerate(records):
                samples[(day % 6, t)].setdefault(slot, []).append(
                    (int(row[3]), max(int(row[2]) - int(row[1]), 1)))

    prototypes = {}
    for key, by_slot in samples.items():
        prototypes[key] = []
        for slot in sorted(by_slot):
            records = by_slot[slot]
            nodes = [x[0] for x in records]
            medoid = min(nodes, key=lambda a: sum(
                distance_calculator.get_straight_line_distance(a, b) for b in nodes
            ))
            slack = int(np.median([x[1] for x in records]))
            prototypes[key].append((int(medoid), slack))
    return prototypes
