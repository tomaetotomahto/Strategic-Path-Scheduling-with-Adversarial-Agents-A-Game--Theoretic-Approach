from __future__ import annotations
import random, networkx as nx
from agents import AdversarialAgent, CooperativeAgent, SelfishAgent
from game import PathSchedulingGame

def grid_network(n, m):
    G = nx.grid_2d_graph(n, m)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
        if not G.is_directed():
            G[v][u]["weight"] = 1
    return G

def small_world_network(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G

def scale_free_network(n):
    G = nx.barabasi_albert_graph(n, 2)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
        if not G.is_directed():
            G[v][u]["weight"] = 1
    return G

random.seed(42)

# Grid demo
print("=== 5x5 Grid Demo (budget=2) ===")
G = grid_network(5, 5)
sa = SelfishAgent((0, 0), (4, 4))
ca = CooperativeAgent((4, 0), (0, 4))
adv = AdversarialAgent(budget=2, deception=True)
game = PathSchedulingGame(G, [sa, ca], adv)
for r in range(1, 6):
    print(f"Round {r}:", game.play_round())

# Small‑world demo
print("\n=== Small-World Demo ===")
SW = small_world_network(20, 4, 0.1)
sa1 = SelfishAgent(0, 10)
sa2 = SelfishAgent(5, 15)
adv_sw = AdversarialAgent(budget=1, deception=False)
print("Outcome:", PathSchedulingGame(SW, [sa1, sa2], adv_sw).play_round())

# Scale‑free demo
print("\n=== Scale-Free Demo ===")
SF = scale_free_network(30)
sa3 = SelfishAgent(0, 25)
sa4 = SelfishAgent(2, 29)
adv_sf = AdversarialAgent(budget=2, deception=True)
print("Outcome:", PathSchedulingGame(SF, [sa3, sa4], adv_sf).play_round())