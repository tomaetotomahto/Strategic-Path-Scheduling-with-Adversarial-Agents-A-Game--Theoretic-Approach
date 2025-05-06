"""Path-finding utilities (A* search with optional blocked-edge filtering)."""
from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, Iterable, List, Tuple
import networkx as nx

Node = Any
Edge = Tuple[Node, Node]
HeuristicFn = Callable[[Node, Node], float]


def astar_shortest_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    *,
    heuristic: HeuristicFn | None = None,
    blocked_edges: Iterable[Edge] | None = None,
) -> Tuple[float, List[Node] | None]:
    """Return (distance, path) using A*; avoid *blocked_edges* if given."""
    blocked = set(blocked_edges) if blocked_edges else set()

    g: Dict[Node, float] = {start: 0.0}
    f: Dict[Node, float] = {start: (heuristic(start, goal) if heuristic else 0.0)}
    pq: List[Tuple[float, Node, List[Node]]] = [(f[start], start, [start])]
    closed: set[Node] = set()

    while pq:
        _, node, path = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            return g[node], path
        for nbr, data in graph[node].items():
            if (node, nbr) in blocked or (not graph.is_directed() and (nbr, node) in blocked):
                continue
            w = data.get("weight", 1)
            tentative = g[node] + w
            if tentative < g.get(nbr, math.inf):
                g[nbr] = tentative
                h = heuristic(nbr, goal) if heuristic else 0.0
                heapq.heappush(pq, (tentative + h, nbr, path + [nbr]))

    return math.inf, None


def manhattan_heuristic(u: Node, v: Node) -> float:
    if isinstance(u, tuple) and isinstance(v, tuple):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])
    return 0.0