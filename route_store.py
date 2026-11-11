import networkx as nx
from typing import Dict, Tuple, List, Union, Optional

class PathCache:
    
    def __init__(self):
        self._path_length_cache: Dict[Tuple[int, int], Union[int, float]] = {}
        self._path_cache: Dict[Tuple[int, int], List[int]] = {}
        
        self._path_length_hits = 0
        self._path_length_misses = 0
        self._path_hits = 0
        self._path_misses = 0
        
    def get_shortest_path_length(self, graph: nx.Graph, start_node: int, end_node: int) -> Union[int, float]:
        cache_key = (start_node, end_node)
        
        if cache_key in self._path_length_cache:
            self._path_length_hits += 1
            return self._path_length_cache[cache_key]
        
        self._path_length_misses += 1
        try:
            path_length = nx.shortest_path_length(graph, start_node, end_node)
            self._path_length_cache[cache_key] = path_length
            return path_length
        except Exception as e:
            error_value = float('inf')
            self._path_length_cache[cache_key] = error_value
            raise e
    
    def get_shortest_path(self, graph: nx.Graph, start_node: int, end_node: int) -> List[int]:
        cache_key = (start_node, end_node)
        
        if cache_key in self._path_cache:
            self._path_hits += 1
            return self._path_cache[cache_key].copy() 
        
        self._path_misses += 1
        try:
            path = nx.shortest_path(graph, start_node, end_node)
            self._path_cache[cache_key] = path
            return path
        except Exception as e:
            empty_path = []
            self._path_cache[cache_key] = empty_path
            raise e
    
    def clear_cache(self):
        self._path_length_cache.clear()
        self._path_cache.clear()
        self._reset_stats()
    
    def _reset_stats(self):
        self._path_length_hits = 0
        self._path_length_misses = 0
        self._path_hits = 0
        self._path_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        total_path_length_requests = self._path_length_hits + self._path_length_misses
        total_path_requests = self._path_hits + self._path_misses
        
        path_length_hit_rate = (self._path_length_hits / total_path_length_requests * 100 
                               if total_path_length_requests > 0 else 0)
        path_hit_rate = (self._path_hits / total_path_requests * 100 
                        if total_path_requests > 0 else 0)
        
        return {
            'path_length_cache_size': len(self._path_length_cache),
            'path_cache_size': len(self._path_cache),
            'path_length_hits': self._path_length_hits,
            'path_length_misses': self._path_length_misses,
            'path_length_hit_rate': path_length_hit_rate,
            'path_hits': self._path_hits,
            'path_misses': self._path_misses,
            'path_hit_rate': path_hit_rate,
            'total_cache_entries': len(self._path_length_cache) + len(self._path_cache)
        }

_global_path_cache = PathCache()

def cached_shortest_path_length(graph: nx.Graph, start_node: int, end_node: int) -> Union[int, float]:
    return _global_path_cache.get_shortest_path_length(graph, start_node, end_node)

def cached_shortest_path(graph: nx.Graph, start_node: int, end_node: int) -> List[int]:
    return _global_path_cache.get_shortest_path(graph, start_node, end_node)

def get_global_cache_stats() -> Dict[str, Union[int, float]]:
    return _global_path_cache.get_cache_stats()

def print_global_cache_stats():
    _global_path_cache.print_stats()

def clear_global_cache():
    _global_path_cache.clear_cache()

if __name__ == "__main__":
    import networkx as nx
    
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    cache = PathCache()
    
