# rlgames

A repository for implementing and experimenting with various reinforcement learning algorithms and environments in Python.

## Implemented Algorithms

### Dynamic Programming

*   **Policy Iteration**: Solves a GridWorld environment by iteratively evaluating and improving a policy.
    *   `dp/policy-iter/pi_gridworld.py`
*   **Value Iteration**: Solves GridWorld and CliffWalking environments by finding the optimal value function.
    *   `dp/value-iter/vi_gridworld.py`
    *   `dp/value-iter/vi_cliff_walking.py`

### Deep Q-Network (DQN)

*   **DQN for Snake**: A Deep Q-Learning agent that learns to play the classic game of Snake.
    *   `dqn-snake/`

### Temporal Difference (TD) Learning

*   **Q-Learning**: An off-policy TD control algorithm used to solve a Maze environment.
    *   `td-learning/q-learning/q-learning.py`
*   **SARSA**: An on-policy TD control algorithm, also used to solve the Maze environment.
    *   `td-learning/sarsa/sarsa.py`

## Environments

*   **GridWorld**: A simple 4x4 grid where the agent's goal is to reach a terminal state.
*   **CliffWalking**: A 4x12 grid where the agent must find a path to the goal while avoiding a cliff.
*   **Snake**: The classic game of Snake, where an agent controls a snake to eat food and grow longer.
*   **Maze**: A 5x5 maze that the agent must navigate to find the goal.

## How to Run

Each algorithm is implemented as a standalone Python script. To run any of the implementations, simply execute the corresponding Python file. For example:

```bash
python dp/policy-iter/pi_gridworld.py
```