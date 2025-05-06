"""Agent definitions - greedy k-edge blocker *without total disconnection*."""
from __future__ import annotations
import math
from typing import Any, List, Sequence, Tuple
import networkx as nx
from pathfinding import astar_shortest_path, manhattan_heuristic

Node = Any
Edge = Tuple[Node, Node]


class BaseAgent:
    def __init__(self, start: Node, goal: Node):
        self.start, self.goal = start, goal

    def choose_path(self, graph: nx.Graph, *, info: dict | None = None):
        raise NotImplementedError


class SelfishAgent(BaseAgent):
    def choose_path(self, graph: nx.Graph, *, info: dict | None = None):
        blocked = {
            edge for edge, cred in info.get("announcements", []) if cred
        } if info else set()
        _, path = astar_shortest_path(graph, self.start, self.goal,
                                      heuristic=manhattan_heuristic,
                                      blocked_edges=blocked)
        return path


class CooperativeAgent(SelfishAgent):
    def choose_path(self, graph: nx.Graph, *, info: dict | None = None):
        blocked = {edge for edge, _ in info.get("announcements", [])} if info else set()
        _, path = astar_shortest_path(graph, self.start, self.goal,
                                      heuristic=manhattan_heuristic,
                                      blocked_edges=blocked)
        return path


class AdversarialAgent:
    """
    Greedy k-edge interdiction.
    """
    def __init__(self, *, budget: int = 2, deception: bool = False):
        self.budget = budget
        self.deception = deception
        self._last_blocked: List[Edge] = []

    # ‑‑ helper ----------------------------------------------------------
    @staticmethod
    def _safe_remove(g: nx.Graph, u: Node, v: Node):
        if g.has_edge(u, v):
            g.remove_edge(u, v)
        if not g.is_directed() and g.has_edge(v, u):
            g.remove_edge(v, u)

    def choose_block_edges(self, graph: nx.Graph, target: BaseAgent | None):
        if self.budget <= 0 or target is None:
            self._last_blocked = []
            return []

        temp = graph.copy()
        blocked: List[Edge] = []

        for _ in range(self.budget):
            # shortest path distance BEFORE blocking
            dist, sp = astar_shortest_path(
                temp, target.start, target.goal, heuristic=manhattan_heuristic
            )
            if not sp or len(sp) < 2:
                break

            best_edge, best_inc = None, 0.0
            for i in range(len(sp) - 1):
                u, v = sp[i], sp[i + 1]
                w = temp[u][v]["weight"]

                # test removal
                self._safe_remove(temp, u, v)
                if nx.has_path(temp, target.start, target.goal):
                    new_dist, _ = astar_shortest_path(
                        temp, target.start, target.goal, heuristic=manhattan_heuristic
                    )
                    inc = new_dist - dist
                    if inc > best_inc:
                        best_inc, best_edge = inc, (u, v)
                # restore edge for next trial
                temp.add_edge(u, v, weight=w)
                if not temp.is_directed():
                    temp.add_edge(v, u, weight=w)

            # fallback: pick first edge that still preserves connectivity
            if best_edge is None:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    self._safe_remove(temp, u, v)
                    if nx.has_path(temp, target.start, target.goal):
                        best_edge = (u, v)
                        break
                    temp.add_edge(u, v, weight=1)
                    if not temp.is_directed():
                        temp.add_edge(v, u, weight=1)

            if best_edge is None:     # cannot block without disconnection
                break

            # commit removal
            u, v = best_edge
            self._safe_remove(temp, u, v)
            blocked.append(best_edge)

        self._last_blocked = blocked
        return blocked

    # ‑‑ announcements ---------------------------------------------------
    def announce_blocks(self, blocked_edges: Sequence[Edge]):
        anns = [(e, True) for e in blocked_edges]
        if self.deception and blocked_edges:
            u, v = blocked_edges[0]
            anns.append(((v, u), False))
        return anns