"""Stackelberg path-scheduling game environment."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple
import networkx as nx
from agents import AdversarialAgent, BaseAgent

Node = Any
Edge = Tuple[Node, Node]


class PathSchedulingGame:
    def __init__(self, graph: nx.Graph, agents: List[BaseAgent],
                 adversary: AdversarialAgent | None = None):
        self.base_graph = graph
        self.graph = graph.copy()
        self.agents = agents
        self.adversary = adversary
        self.trust: Dict[BaseAgent, float] = {a: 1.0 for a in agents}

    def play_round(self):
        actual_blocks, announcements = [], []
        if self.adversary:
            target = self.agents[0] if self.agents else None
            actual_blocks = self.adversary.choose_block_edges(self.graph, target)
            for u, v in actual_blocks:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                if not self.graph.is_directed() and self.graph.has_edge(v, u):
                    self.graph.remove_edge(v, u)
            announcements = self.adversary.announce_blocks(actual_blocks)

        paths, costs = [], []
        for agent in self.agents:
            info = {"announcements": [
                (edge, True if truth else self.trust[agent] >= 0.5)
                for edge, truth in announcements
            ]}
            path = agent.choose_path(self.graph, info=info)
            paths.append(path)
            cost = 0.0
            if path is None:
                cost = math.inf
            else:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if (u, v) in actual_blocks or (
                        not self.graph.is_directed() and (v, u) in actual_blocks
                    ):
                        cost = math.inf
                        break
                    cost += self.base_graph[u][v]["weight"]
            costs.append(cost)

        # Trust update
        for _, truth in announcements:
            for agent in self.agents:
                self.trust[agent] = max(0.0, min(1.0,
                    self.trust[agent] + (0.1 if truth else -0.3)))

        return {
            "paths": paths,
            "costs": costs,
            "blocks": actual_blocks,
            "announcements": announcements,
        }