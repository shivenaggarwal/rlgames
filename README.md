# rlgames

> **[!note]**  
> this is just a toy repository for learning reinforcement learning stuff

some reinforcement learning algorithms and environments i've implemented in python. nothing fancy, just code to understand how different rl methods work.

## algorithms

### dynamic programming
solving problems when you know the environment model.

- **policy iteration** - improves policies step by step in gridworld
  - [`dp/policy-iter/pi_gridworld.py`](dp/policy-iter/pi_gridworld.py)
- **value iteration** - finds optimal values for each state
  - [`dp/value-iter/vi_gridworld.py`](dp/value-iter/vi_gridworld.py)
  - [`dp/value-iter/vi_cliff_walking.py`](dp/value-iter/vi_cliff_walking.py)

### deep q-network (dqn)
using neural networks for q-learning.

- **dqn for snake** - neural network learns to play snake
  - [`dqn-snake/`](dqn-snake/)

### temporal difference learning
learning directly from experience without knowing the environment.

- **q-learning** - learns optimal actions for each state
  - [`td-learning/q-learning/q-learning.py`](td-learning/q-learning/q-learning.py)
- **sarsa** - learns while following a specific policy
  - [`td-learning/sarsa/sarsa.py`](td-learning/sarsa/sarsa.py)

### function approximation
handling continuous state spaces with linear approximation.

- **linear function approximation** - sarsa with tile coding for mountaincar
  - [`function-approximation/linear/mountain-car.py`](function-approximation/linear/mountain-car.py)

## environments

| environment | description | difficulty |
|-------------|-------------|------------|
| gridworld | simple 4x4 grid navigation | easy |
| cliffwalking | 4x12 grid with dangerous cliff edge | medium |
| snake | classic snake game | hard |
| maze | 5x5 maze navigation | medium |
| mountaincar | get car up hill with continuous states | medium |

## running the code

just run any python file directly:

```bash
python dp/policy-iter/pi_gridworld.py
python dqn-snake/main.py
python td-learning/q-learning/q-learning.py
python function-approximation/linear/mountain-car.py
```