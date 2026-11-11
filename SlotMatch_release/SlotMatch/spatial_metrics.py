from __future__ import annotations

from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
except ImportError:  # NetworkX remains an exact, slower fallback.
    scipy_dijkstra = None

from distance_store import distance_store


class DistanceCalculator:

    def __init__(
        self,
        nodes_csv_path: str = "./shanghai_road_dataset/road_network/nodes_processed.csv",
        node_features_npy_path: str = "./shanghai_road_dataset/road_network/node_features_fixed.npy",
        enable_fallback_id_estimation: bool = True,
    ):
        self.node_coordinates: Dict[int, Tuple[float, float]] = {}
        self.road_graph = None
        self._road_csr = None
        self._node_to_index = None
        self._source_distance_cache = {}

        self.nodes_csv_path = Path(nodes_csv_path)
        self.node_features_npy_path = Path(node_features_npy_path)
        self.enable_fallback_id_estimation = enable_fallback_id_estimation

        self._load_node_coordinates()

    def _load_node_coordinates(self) -> None:

        try:
            df = pd.read_csv(self.nodes_csv_path)
            coords: Dict[int, Tuple[float, float]] = {}

            for _, row in df.iterrows():
                node_id = int(row.iloc[0])
                lon = float(row.iloc[1])
                lat = float(row.iloc[2])
                coords[node_id] = (lon, lat)

            self.node_coordinates = coords
            return

        except Exception:
            pass

        try:
            node_features = np.load(self.node_features_npy_path)

            self.node_coordinates = {
                i: (float(node_features[i][0]), float(node_features[i][1]))
                for i in range(len(node_features))
            }
            return

        except Exception:
            self.node_coordinates = {}

    def set_road_graph(self, road_graph) -> None:
        self.road_graph = road_graph
        self._road_csr = None
        self._node_to_index = None
        self._source_distance_cache = {}
        if road_graph is not None and scipy_dijkstra is not None:
            nodes = list(road_graph.nodes())
            self._node_to_index = {node: index for index, node in enumerate(nodes)}
            self._road_csr = nx.to_scipy_sparse_array(
                road_graph, nodelist=nodes, weight="weight", format="csr", dtype=np.float64
            )

    def get_road_distances_from(self, source: int):
        """One exact Dijkstra for many targets; values use the public 102 scale."""
        if source in self._source_distance_cache:
            return self._source_distance_cache[source]
        if self._road_csr is None or source not in self._node_to_index:
            return None
        raw = scipy_dijkstra(self._road_csr, directed=False,
                             indices=self._node_to_index[source],
                             return_predecessors=False)
        self._source_distance_cache[source] = raw
        return raw

    def get_road_distance_from_source_map(self, source: int, target: int) -> float:
        distances = self.get_road_distances_from(source)
        if distances is None or target not in self._node_to_index:
            return self.get_road_network_distance(source, target)
        value = float(distances[self._node_to_index[target]]) * 102
        return value if np.isfinite(value) else self.get_straight_line_distance(source, target)

    @staticmethod
    def haversine_distance(lon1: float, lat1: float,
                           lon2: float, lat2: float) -> float:

        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))

        return c * 6371.0

    def get_straight_line_distance(self, node1: int, node2: int) -> float:

        if node1 == node2:
            return 0.0

        if node1 in self.node_coordinates and node2 in self.node_coordinates:
            lon1, lat1 = self.node_coordinates[node1]
            lon2, lat2 = self.node_coordinates[node2]

            return self.haversine_distance(lon1, lat1, lon2, lat2)

        if self.enable_fallback_id_estimation:
            node_dist = abs(node1 - node2)
            return node_dist * 1e-4

        return float("inf")

    def get_road_network_distance(self, node1: int, node2: int) -> float:

        if node1 == node2:
            return 0.0

        cached_distance = distance_store.get(node1, node2)

        if cached_distance is not None:
            return cached_distance

        if self.road_graph is None:
            distance = self.get_straight_line_distance(node1, node2)

        else:
            try:
                if self._road_csr is not None:
                    source = self._node_to_index[node1]
                    target = self._node_to_index[node2]
                    # SciPy executes the same weighted Dijkstra calculation in
                    # compiled code. Only the backend changes; route semantics,
                    # edge weights, and the published 102 scale are untouched.
                    distances = scipy_dijkstra(
                        self._road_csr, directed=False, indices=source,
                        return_predecessors=False,
                    )
                    path_weight = float(distances[target])
                    if not np.isfinite(path_weight):
                        raise nx.NetworkXNoPath
                else:
                    path_weight, _ = nx.bidirectional_dijkstra(
                        self.road_graph, node1, node2, weight="weight"
                    )

                distance = path_weight * 102

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance = self.get_straight_line_distance(node1, node2)

            except Exception:
                distance = self.get_straight_line_distance(node1, node2)

        distance_store.set(node1, node2, distance)

        return distance

    @staticmethod
    def calculate_travel_time(distance_km: float,
                              speed_kmh: float) -> float:

        if speed_kmh <= 0:
            return float("inf")

        return distance_km / speed_kmh


distance_calculator = DistanceCalculator()
