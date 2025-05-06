# Strategic Path Scheduling with Adversarial Agents

A game-theoretic framework and open-source simulation for routing in contested networks, where malicious agents block or lie to maximize disruption.  
(CSE 556: Game Theory with Applications to Networks, ASU)

---

## Overview

Traditional path-finding assumes all agents minimize travel time—but in many real systems, some players act *adversarially*, blocking critical links or spreading false information to slow competitors.  We model these scenarios as a **Bayesian Stackelberg routing game**:

- **Leaders** (adversaries) commit to blocking up to *k* edges (with a connectivity guard) and optional deception.  
- **Followers** (selfish or cooperative agents) observe signals and re-plan shortest paths under congestion.  

Key contributions:  
1. **Bi-level optimization model** extending Dobss MILP with a fast greedy fallback.  
2. **Connectivity-guarded adversary** so networks stay connected (no infinite-cost traps).  
3. **Hybrid counter-strategies** (APS, IS, TBR) reducing adversarial damage by 60–72 %.  
4. **Open-source Python/NetworkX test-bed** with unit tests and reproducible demos.

For full paper, see [Game Theory Project PDF](Game_Theory_Project.pdf).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/tomaetotomahto/Strategic-Path-Scheduling-with-Adversarial-Agents-A-Game--Theoretic-Approach.git
cd Strategic-Path-Scheduling

# 2. Install
pip install networkx

# 3. Run demos
python simulation.py
# --> 5×5 grid (budget=2), small-world, scale-free

# 4. Run tests
python -m unittest discover tests
