"""Bounded memoization of identical SciPy calls; no distance approximation."""
from collections import OrderedDict


def memoize_dijkstra(original, max_sources=128):
    graph_ref = None
    cache = OrderedDict()

    def cached(graph, *, directed, indices, return_predecessors):
        nonlocal graph_ref
        if graph is not graph_ref:
            graph_ref = graph
            cache.clear()
        key = (bool(directed), int(indices), bool(return_predecessors))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        answer = original(graph, directed=directed, indices=indices,
                          return_predecessors=return_predecessors)
        cache[key] = answer
        if len(cache) > max_sources:
            cache.popitem(last=False)
        return answer

    return cached
