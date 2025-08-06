import numpy as np
import matplotlib.pyplot as plt

# represents our mdp env where our agent will learn
# a 4x4 gridworld env

class GridWorld:
    def __init__(self):
        self.size = 4  # 4x4 grid world (16 states)
        self.num_states = self.size * self.size  

        # actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.num_actions = len(self.actions)

        # terminal states(top-left and bottom-right)
        self.terminal_states = [(0, 0), (3, 3)]  

        # action directions
        self.action_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def state_to_index(self, state): 
        # convert (row, col) to linear index
        # eg. state(1, 2) -> index 1*4 + 2 = 6
        return state[0] * self.size + state[1]

    def index_to_state(self, index):  
        # converts linear index to (row col)
        return (index // self.size, index % self.size)

    def get_next_state(self, state, action): 
        # get next state given current state and action
        if state in self.terminal_states:
            return state  # terminal state stays put

        row, col = state
        d_row, d_col = self.action_dirs[action]

        # calculate the new position
        new_row = max(0, min(self.size - 1, row + d_row))
        new_col = max(0, min(self.size - 1, col + d_col))

        return (new_row, new_col)

    def get_reward(self, state, action, next_state): 
        # get the reward for the transiton
        if state in self.terminal_states:
            return 0.0
        return -1.0  # small negative reward for each step


# pi algo
class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma  # discount factor
        self.theta = theta  # convergence threshold

        # initialize random policy uniform over actions
        # basically created a 16x4 matrix where each row represents a state and each column an action
        # initially each has a prob of 0.25
        self.policy = np.ones((env.num_states, env.num_actions)) / env.num_actions

        # init value function
        self.V = np.zeros(env.num_states)

        # store history for visualization
        self.value_history = []
        self.policy_history = []

    # eval current policy using iterative policy evaluation
    def policy_evaluation(self):
        print("  Policy Evaluation...")
        iteration = 0

        while True:
            delta = 0.0  # tracks max change in any state's value this iteration
            old_V = self.V.copy()  # stores previous values for comparison

            # update value for each state
            for s_idx in range(self.env.num_states):
                state = self.env.index_to_state(s_idx)

                # skip the terminal_states
                if state in self.env.terminal_states:
                    continue

                # store the current value before updating
                v_old = self.V[s_idx]

                # calculate the new value using bellman equation
                v_new = 0.0
                for a in self.env.actions:

                    # get the π(a | s) prob of taking that action a in state s
                    action_prob = self.policy[s_idx, a]

                    # get the next state and reward
                    next_state = self.env.get_next_state(state, a)  # find where we will end up
                    reward = self.env.get_reward(state, a, next_state)  # get immediate reward
                    # convert the next state to index for array lookup
                    next_s_idx = self.env.state_to_index(next_state)

                    # add to value (transition probability is 1.0 for deterministic env)
                    v_new += action_prob * (reward + self.gamma * self.V[next_s_idx])

                self.V[s_idx] = v_new
                delta = max(delta, abs(v_old - v_new))

            iteration += 1
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}, Delta: {delta:.6f}")  

            # check for convergence
            if delta < self.theta:
                print(f"    Converged after {iteration} iterations")
                break

    def policy_improvement(self):
        # improve policy using current value function
        print("  Policy Improvement...")
        policy_stable = True  # tracks whether we made any changes to the policy

        for s_idx in range(self.env.num_states):
            state = self.env.index_to_state(s_idx)  

            # skip terminal states
            if state in self.env.terminal_states:
                continue

            # store old action probabilities
            old_policy = self.policy[s_idx].copy()

            # calculate action values (Q-values)
            action_values = np.zeros(self.env.num_actions)
            for a in self.env.actions:
                next_state = self.env.get_next_state(state, a)
                reward = self.env.get_reward(state, a, next_state)
                next_s_idx = self.env.state_to_index(next_state)

                action_values[a] = reward + self.gamma * self.V[next_s_idx]

            # find best action(s)
            max_value = np.max(action_values)
            best_actions = np.where(np.abs(action_values - max_value) < 1e-10)[0]

            # create new greedy policy (uniform over best actions)
            new_policy = np.zeros(self.env.num_actions)
            new_policy[best_actions] = 1.0 / len(best_actions)

            # update policy
            self.policy[s_idx] = new_policy

            # check if policy changed
            if not np.array_equal(old_policy, new_policy):
                policy_stable = False

        return policy_stable

    # run the complete policy iteration algorithm
    def run_policy_iteration(self, max_iterations=100):
        print("Starting Policy Iteration...")
        print(f"Environment: {self.env.size}x{self.env.size} GridWorld")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        print("-" * 50)

        for iteration in range(max_iterations):
            print(f"Policy Iteration {iteration + 1}:")

            # store current state for history
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.get_deterministic_policy().copy())

            # step 1: Policy Evaluation
            self.policy_evaluation()

            # step 2: Policy Improvement
            policy_stable = self.policy_improvement()

            print(f"  Policy stable: {policy_stable}")
            print()

            # if policy is stable, we've found the optimal policy
            if policy_stable:
                print(f"Optimal policy found after {iteration + 1} iterations!")
                break

        # store final state
        self.value_history.append(self.V.copy())
        self.policy_history.append(self.get_deterministic_policy().copy())

    # get deterministic policy (best action for each state)
    def get_deterministic_policy(self):
        det_policy = np.zeros(self.env.num_states, dtype=int)
        for s_idx in range(self.env.num_states):
            det_policy[s_idx] = np.argmax(self.policy[s_idx])
        return det_policy

    # print value function as a grid
    def print_value_function(self):
        print("Value Function:")
        for i in range(self.env.size):
            for j in range(self.env.size):
                s_idx = self.env.state_to_index((i, j))
                print(f"{self.V[s_idx]:8.2f}", end="")
            print()
        print()

    # print policy as a grid with arrows
    def print_policy(self):
        action_symbols = ['↑', '→', '↓', '←']
        print("Policy:")
        det_policy = self.get_deterministic_policy()

        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                s_idx = self.env.state_to_index(state)
                
                if state in self.env.terminal_states:
                    print("  T  ", end="")
                else:
                    action = det_policy[s_idx]
                    print(f"  {action_symbols[action]}  ", end="")
            print()
        print()


if __name__ == "__main__":
    # create environment and run policy iteration
    env = GridWorld()
    pi = PolicyIteration(env, gamma=0.9, theta=1e-6)  
    print("Initial Policy:")
    pi.print_policy()
    print("Initial Value Function:")
    pi.print_value_function()

    # run policy iteration
    pi.run_policy_iteration()

    print("Final Results:")
    print("=" * 50)
    pi.print_value_function()
    pi.print_policy()
