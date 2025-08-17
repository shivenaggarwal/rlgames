# rl algorithms 

> [!NOTE]
> this is just a toy repository for learning reinforcement learning stuff

# reinforcement learning algorithms

a collection of reinforcement learning algorithms implemented in python for learning purposes.

## what's inside?

this repository contains implementations of several fundamental reinforcement learning algorithms, including:

*   **dynamic programming:**
    *   policy iteration
    *   value iteration
*   **temporal difference (td) learning:**
    *   q-learning
    *   sarsa
*   **deep q-networks (dqn):**
    *   dqn for the game of snake
*   **policy gradient methods:**
    *   reinforce
*   **function approximation:**
    *   linear function approximation with tile coding
    *   non-linear function approximation with neural networks

## algorithms

### dynamic programming

dynamic programming methods are used when the environment's model is fully known.

*   **policy iteration:** iteratively improves the policy and value function until an optimal policy is found.
    *   implementation: [`dp/policy-iter/pi_gridworld.py`](dp/policy-iter/pi_gridworld.py)
*   **value iteration:** finds the optimal value function by iteratively updating the value of each state.
    *   implementations:
        *   [`dp/value-iter/vi_gridworld.py`](dp/value-iter/vi_gridworld.py)
        *   [`dp/value-iter/vi_cliff_walking.py`](dp/value-iter/vi_cliff_walking.py)

### temporal difference (td) learning

td learning methods learn directly from raw experience without a model of the environment's dynamics.

*   **q-learning:** an off-policy td control algorithm that learns the optimal action-value function.
    *   implementation: [`td-learning/q-learning/q-learning.py`](td-learning/q-learning/q-learning.py)
*   **sarsa:** an on-policy td control algorithm that learns the action-value function relative to the policy being followed.
    *   implementation: [`td-learning/sarsa/sarsa.py`](td-learning/sarsa/sarsa.py)

### deep q-networks (dqn)

dqn uses a deep neural network to approximate the q-value function, making it suitable for high-dimensional state spaces.

*   **dqn for snake:** a dqn agent that learns to play the classic game of snake.
    *   implementation: [`dqn-snake/`](dqn-snake/)

### policy gradient methods

policy gradient methods directly optimize the policy by performing gradient ascent on the expected return.

*   **reinforce:** a monte carlo policy gradient algorithm that updates the policy based on the return of entire episodes.
    *   implementation: [`policy-gradients/reinforce.py`](policy-gradients/reinforce.py)

### function approximation

function approximation is used to handle large or continuous state spaces by approximating the value function or policy.

*   **linear function approximation:** uses tile coding to approximate the value function for the mountain car environment.
    *   implementation: [`function-approximation/linear/mountain-car.py`](function-approximation/linear/mountain-car.py)
*   **non-linear function approximation:** uses a neural network to approximate the value function.
    *   implementation: [`function-approximation/non-linear/mountain-car.py`](function-approximation/non-linear/mountain-car.py)

## environments

| environment     | description                               | implemented in                                      |
| --------------- | ----------------------------------------- | --------------------------------------------------- |
| gridworld       | a simple 4x4 grid for navigation.         | `dp/`                                               |
| cliff walking   | a 4x12 grid with a "cliff" to avoid.      | `dp/value-iter/vi_cliff_walking.py`                 |
| maze            | a 5x5 maze for navigation.                | `td-learning/`                                      |
| snake           | the classic game of snake.                | `dqn-snake/`                                        |
| mountain car    | a classic control problem.                | `function-approximation/`                           |
| cartpole        | a classic control problem.                | `policy-gradients/`                                 |

## how to run

to run any of the algorithms, simply execute the corresponding python script:

```bash
# dynamic programming
python dp/policy-iter/pi_gridworld.py
python dp/value-iter/vi_gridworld.py

# td learning
python td-learning/q-learning/q-learning.py
python td-learning/sarsa/sarsa.py

# dqn
python dqn-snake/agent.py

# policy gradients
python policy-gradients/reinforce.py

# function approximation
python function-approximation/linear/mountain-car.py
python function-approximation/non-linear/mountain-car.py
```

## dependencies

the following libraries are required to run the code:

*   `numpy`
*   `matplotlib`
*   `torch`
*   `gymnasium`
*   `pygame`
