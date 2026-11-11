#!/usr/bin/env python3
import os
import pickle
import atexit
from typing import Dict, Tuple, Optional


class RoadDistanceCache:

    def __init__(self, cache_file: str = "./cache/distance_store.pkl"):
        self.cache_file = cache_file
        self.cache: Dict[Tuple[int, int], float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.dirty_count = 0
        self._load_cache()
        atexit.register(self.save_cache)

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self) -> None:
        try:
            parent = os.path.dirname(self.cache_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            return

    def get(self, from_node: int, to_node: int) -> Optional[float]:
        if from_node == to_node:
            return 0.0

        key = (min(from_node, to_node), max(from_node, to_node))
        val = self.cache.get(key)

        if val is not None:
            self.hit_count += 1
            return val

        self.miss_count += 1
        return None

    def set(self, from_node: int, to_node: int, distance: float) -> None:
        if from_node == to_node:
            return

        key = (min(from_node, to_node), max(from_node, to_node))
        self.cache[key] = float(distance)
        self.dirty_count += 1
        if self.dirty_count >= 5000:
            self.save_cache()
            self.dirty_count = 0

    def get_or_compute(self, from_node: int, to_node: int, compute_func) -> float:
        cached = self.get(from_node, to_node)
        if cached is not None:
            return cached

        distance = float(compute_func(from_node, to_node))
        self.set(from_node, to_node, distance)
        return distance

    def get_stats(self) -> Dict[str, object]:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }


distance_store = RoadDistanceCache()
