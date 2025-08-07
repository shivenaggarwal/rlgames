import numpy as np
import matplotlib as plt


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


# vi algo
class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma  # discount factor
        self.theta = theta  # convergenve thershold

        # initialize value function to zeros
        self.V = np.zeros(env.num_states)

        # store the history for visualization
        self.value_history = []

    def value_iteration_step(self):  # single step of value iteration using boe
        delta = 0.0  # track maximum change across all the num_states
        # if we update self.V directly we would use some old and some new values in the same iteration which would be incorrect
        new_V = np.zeros_like(self.V)  # create a new Value array

        # update value for each state
        for s_idx in range(self.env.num_states):
            state = self.env.index_to_state(s_idx)
            # terminal states have value 0
            if state in self.env.terminal_states:
                new_V[s_idx] = 0.0
                continue
            # store old value
            v_old = self.V[s_idx]

            # calculate Q-values for all actions and take maximum
            action_values = []
            for a in self.env.actions:
                # get next state and reward for this action
                next_state = self.env.get_next_state(state, a)  # where we will end up
                reward = self.env.get_reward(
                    state, a, next_state
                )  # get immediate reward
                # convert to index for array lookup
                next_s_idx = self.env.state_to_index(next_state)

                # calculate Q(s,a) = R(s,a,s') + gamme * V(s')
                # immediate reward for taking the action and discounted future value from where we end up
                q_value = reward + self.gamma * self.V[next_s_idx]
                action_values.append(q_value)

            # bellman optimality equation: V(s) = max_a Q(s,a)
            # instead of following a specific policy, we assume we'll always take the best action. So V(s) equals the value of the best action
            new_V[s_idx] = max(action_values)

            # track maximum change
            delta = max(delta, abs(v_old - new_V[s_idx]))

        # update value function
        self.V = new_V
        return delta

    # main algo loop
    def run_value_iteration(self, max_iterations=1000):
        print("Starting Value Iteration...")
        print(f"Environment: {self.env.size}x{self.env.size} GridWorld")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        print("-" * 50)

        for iteration in range(max_iterations):
            # store current values for history
            self.value_history.append(self.V.copy())

            # perform one step of value iteration
            delta = self.value_iteration_step()

            # print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}, Delta: {delta:.6f}")

            # check for convergence
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations")
                print(f"Final delta: {delta:.6f}")
                break

        else:
            print(f"Reached maximum iterations ({max_iterations})")

        # store final values
        self.value_history.append(self.V.copy())

    # extract optimal policy from optimal value function
    def extract_policy(self):
        policy = np.zeros(self.env.num_states, dtype=int)
        for s_idx in range(self.env.num_states):
            state = self.env.index_to_state(s_idx)

            # terminal states don't need actions
            if state in self.env.terminal_states:
                policy[s_idx] = 0  # arbitrary won't be used
                continue

            # calculate Q-values for all actions
            action_values = []
            for a in self.env.actions:
                next_state = self.env.get_next_state(state, a)
                reward = self.env.get_reward(state, a, next_state)
                next_s_idx = self.env.state_to_index(next_state)
                
                q_value = reward + self.gamma * self.V[next_s_idx]
                action_values.append(q_value)

            # choose action with maximum Q-value
            policy[s_idx] = np.argmax(action_values)

        return policy

    def print_value_function(self):
        print("Value Function:")
        for i in range(self.env.size):
            for j in range(self.env.size):
                s_idx = self.env.state_to_index((i, j))
                print(f"{self.V[s_idx]:8.2f}", end="")
            print()
        print()

    
    def print_policy(self):
        action_symbols = ['↑', '→', '↓', '←']
        print("Policy:")
        policy = self.extract_policy()
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                s_idx = self.env.state_to_index(state)
                
                if state in self.env.terminal_states:
                    print("  T  ", end="")
                else:
                    action = policy[s_idx]
                    print(f"  {action_symbols[action]}  ", end="")
            print()
        print() 



if __name__ == "__main__":
    # Run Value Iteration
    env = GridWorld()
    vi = ValueIteration(env, gamma=0.9, theta=1e-6)
    
    print("Initial Value Function:")
    vi.print_value_function()
    
    # Run the algorithm
    vi.run_value_iteration()
    
    print("\nFinal Results:")
    print("=" * 50)
    vi.print_value_function()
    vi.print_policy()
