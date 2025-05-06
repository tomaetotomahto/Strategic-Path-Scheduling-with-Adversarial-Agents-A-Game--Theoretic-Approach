import unittest, networkx as nx
from agents import SelfishAgent, CooperativeAgent, AdversarialAgent
from game import PathSchedulingGame


class GridInterdictionTest(unittest.TestCase):
    """Basic sanity: adversary should block ≥ 1 edge yet keep the grid connected."""

    def setUp(self):
        G = nx.grid_2d_graph(5, 5)
        for u, v in G.edges():
            G[u][v]["weight"] = 1
        self.game = PathSchedulingGame(
            G,
            [
                SelfishAgent((0, 0), (4, 4)),
                CooperativeAgent((4, 0), (0, 4)),
            ],
            AdversarialAgent(budget=2, deception=False),
        )
        self.result = self.game.play_round()

    def test_blocks_non_empty(self):
        self.assertGreaterEqual(len(self.result["blocks"]), 1)

    def test_paths_finite(self):
        # Both agents should still reach the goal (no inf cost).
        for cost in self.result["costs"]:
            self.assertNotEqual(cost, float("inf"))


if __name__ == "__main__":
    unittest.main()