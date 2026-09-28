# Reinforcement Learning in Bomberman

This repository contains two reinforcement learning agents developed for the Machine Learning Essentials Bomberman tournament. The agents must navigate a 17x17 grid, drop delayed-fuse bombs to destroy crates, collect coins, and trap up to three opposing agents without committing suicide.

## Agent Architectures

### 1. Tabular Dyna-Q+ with $D_4$ Symmetry (Primary Agent)
Our primary, tournament-submitted agent utilizes a model-based Tabular Dyna-Q+ architecture that combines real-world learning with simulated planning. 
* **State Space Canonicalization:** The environment is encoded into a discrete 9-tuple feature vector representing 97,200 possible states. We apply $D_4$ dihedral group symmetry (4 rotations, 2 reflections) to canonicalize the board orientation, reducing the effective exploration space by a factor of 8.
* **Simulated Planning:** For every real step taken, the agent performs 20 simulated planning backups from a learned, count-based stochastic transition model.
* **Exploration Bonus:** To prevent deadlock, the agent applies an exploration bonus of $\kappa\sqrt{\tau}$ during planning to encourage revisiting stale states.
* **Dynamic Target Hierarchy:** The agent actively hunts by prioritizing trap kill positions within 4 moves (where a bomb leaves the opponent 0 escape routes), followed by coins, then crates and opponents.

### 2. Linear Function Approximation (Baseline Agent)
An exploratory model-free baseline agent that approximates Q-values using 4 weights tied to 4 binary features: bombing near crates, bombing near enemies, matching a BFS path to a target, and closing distance to opponents.
* **Safety Filter:** Relies on a hardcoded 6-step temporal hazard lookahead to strip suicidal moves from the candidate action list before applying the $\epsilon$-greedy policy.
* **Learning:** Performs a 1-step Temporal Difference update per real step with a constant $\epsilon=0.15$ exploration rate. 

## Tournament Performance

Agents were evaluated under strict tournament conditions: a 0.5-second decision limit, $\epsilon = 0.01$ greedy policy, and 400 full rounds across 10 fixed seeds.

| Agent | Score per Round | Coins per Round | Kills per Round | Suicide Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Dyna-Q+ (v10)** | **3.47** | 2.50 | 0.19 | 0.55 |
| `rule_based_agent` | 3.05 | 2.22 | 0.19 | - |
| **Baseline_Agent** | 2.75 | 2.40 | 0.07 | 0.62 |

The Tabular Dyna-Q+ agent significantly outperformed the course's `rule_based_agent` ($t = +2.74$, $p < 0.05$) while maintaining a strict coin advantage ($t = +3.74$).

## Repository Structure

* `dynaq_agent/`: Contains the primary model-based agent, including the core Dyna-Q+ engine (`dyna.py`), mixed-radix state encoding (`features.py`), and the fully trained `dynaq_plus_model.pt` checkpoint.
* `Baseline_Agent/`: Contains the baseline linear approximation agent and its trained `my-saved-model.pt` weights.


## Local Execution

To watch the primary agent play against three rule-based opponents, ensure you have the required dependencies (`pygame`, `tqdm`, `numpy`, `scipy`, `sklearn`) and run:

```bash
python main.py play --agents dynaq_agent rule_based_agent rule_based_agent rule_based_agent

