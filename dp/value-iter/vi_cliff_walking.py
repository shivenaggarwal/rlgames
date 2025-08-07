import numpy as np
import matplotlib.pyplot as plt


class CliffWalkingEnvironment:
    """
    cliff walking environment:
    - 4x12 grid world
    - start at bottom left (3,0)
    - goal at bottom right (3,11)
    - cliff along bottom row (3,1) to (3,10) - falling gives -100 reward and resets to start
    - normal moves give -1 reward
    - reaching goal gives 0 reward and terminates
    """

    def __init__(self):
        self.height = 4
        self.width = 12
        self.num_states = self.height * self.width

        # start and goal positions
        self.start_state = (3, 0)
        self.goal_state = (3, 11)

        # cliff positions (dangerous)
        # bottom row except start/goal
        self.cliff_states = [(3, j) for j in range(1, 11)]

        # actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.num_actions = len(self.actions)
        self.action_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Up', 'Right', 'Down', 'Left']

    # convert (row, col) to linear index

    def state_to_index(self, state):
        return state[0] * self.width + state[1]

    # convert linear index to (row, col)

    def index_to_state(self, index):
        return (index // self.width, index % self.width)

    # check if state is within bounds

    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.height and 0 <= col < self.width

   # get next state given current state and action
    def get_next_state(self, state, action):
        # goal state is terminal
        if state == self.goal_state:
            return state

        row, col = state
        d_row, d_col = self.action_dirs[action]

        # try to move
        new_row = row + d_row
        new_col = col + d_col
        new_state = (new_row, new_col)

        # if move is out of bounds stay in current state
        if not self.is_valid_state(new_state):
            return state

        return new_state

    # get reward for transition
    def get_reward(self, state, action, next_state):
        # already at goal, no reward
        if state == self.goal_state:
            return 0.0

        # fell into cliff big negative reward!
        if next_state in self.cliff_states:
            return -100.0

        # reached goal no additional reward (just end episode)
        if next_state == self.goal_state:
            return 0.0

        # normal move small negative reward (encourages shorter paths)
        return -1.0
   # if agent fell into cliff, reset to start

    def reset_if_cliff(self, state):
        if state in self.cliff_states:
            return self.start_state
        return state

   # print a visual representation of the environment
    def print_environment(self):
        print("Cliff Walking Environment:")
        print("S = Start, G = Goal, C = Cliff, . = Safe")
        print("-" * (self.width * 2 + 1))

        for i in range(self.height):
            print("|", end="")
            for j in range(self.width):
                state = (i, j)
                if state == self.start_state:
                    print("S", end="|")
                elif state == self.goal_state:
                    print("G", end="|")
                elif state in self.cliff_states:
                    print("C", end="|")
                else:
                    print(".", end="|")
            print()
        print("-" * (self.width * 2 + 1))


# vi algo
class ValueIterationCliff:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # initialize value function
        self.V = np.zeros(env.num_states)

        # history for visualization
        self.value_history = []

    # single step of value iteration
    def value_iteration_step(self):
        delta = 0.0
        new_V = np.zeros_like(self.V)
        
        for s_idx in range(self.env.num_states):
            state = self.env.index_to_state(s_idx)
            
            # goal state has value 0
            if state == self.env.goal_state:
                new_V[s_idx] = 0.0
                continue
                
            v_old = self.V[s_idx]
            
            # calculate Q-values for all actions
            action_values = []
            for a in self.env.actions:
                # get immediate next state
                immediate_next = self.env.get_next_state(state, a)
                reward = self.env.get_reward(state, a, immediate_next)
                
                # handle cliff reset: if we fall into cliff we get reset to start
                actual_next = self.env.reset_if_cliff(immediate_next)
                next_s_idx = self.env.state_to_index(actual_next)
                
                # Q(s,a) = R(s,a) + gamma * V(s')
                q_value = reward + self.gamma * self.V[next_s_idx]
                action_values.append(q_value)
            
            # bellman optimality V(s) = max_a Q(s,a)
            new_V[s_idx] = max(action_values)
            delta = max(delta, abs(v_old - new_V[s_idx]))
        
        self.V = new_V
        return delta
   
   # run complete value iteration algorithm
    def run_value_iteration(self, max_iterations=1000):
        print("Starting Value Iteration on Cliff Walking...")
        print(f"Environment: {self.env.height}x{self.env.width} grid")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        print("-" * 50)
        
        for iteration in range(max_iterations):
            self.value_history.append(self.V.copy())
            
            delta = self.value_iteration_step()
            
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}, Delta: {delta:.6f}")
            
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations")
                print(f"Final delta: {delta:.6f}")
                break
        else:
            print(f"Reached maximum iterations ({max_iterations})")
            
        self.value_history.append(self.V.copy())
   # extract optimal policy from value function 
    def extract_policy(self):
        policy = np.zeros(self.env.num_states, dtype=int)
        
        for s_idx in range(self.env.num_states):
            state = self.env.index_to_state(s_idx)
            
            if state == self.env.goal_state:
                policy[s_idx] = 0  # Arbitrary
                continue
            
            # calculate Q-values
            action_values = []
            for a in self.env.actions:
                immediate_next = self.env.get_next_state(state, a)
                reward = self.env.get_reward(state, a, immediate_next)
                actual_next = self.env.reset_if_cliff(immediate_next)
                next_s_idx = self.env.state_to_index(actual_next)
                
                q_value = reward + self.gamma * self.V[next_s_idx]
                action_values.append(q_value)
            
            policy[s_idx] = np.argmax(action_values)
        
        return policy


   #print value function as grid 
    def print_value_function(self):
        print("Value Function:")
        for i in range(self.env.height):
            for j in range(self.env.width):
                s_idx = self.env.state_to_index((i, j))
                print(f"{self.V[s_idx]:7.1f}", end="")
            print()
        print()
   
   # print policy with arrows and special symbols
    def print_policy(self):
        action_symbols = ['↑', '→', '↓', '←']
        print("Optimal Policy:")
        policy = self.extract_policy()
        
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                s_idx = self.env.state_to_index(state)
                
                if state == self.env.start_state:
                    print("  S  ", end="")
                elif state == self.env.goal_state:
                    print("  G  ", end="")
                elif state in self.env.cliff_states:
                    print("  C  ", end="")
                else:
                    action = policy[s_idx]
                    print(f"  {action_symbols[action]}  ", end="")
            print()
        print()
    
    # simulate one episode following the optimal policy
    def simulate_episode(self, max_steps=100):
        policy = self.extract_policy()
        current_state = self.env.start_state
        path = [current_state]
        total_reward = 0.0
        
        for step in range(max_steps):
            if current_state == self.env.goal_state:
                print(f"Reached goal in {step} steps!")
                break
                
            # get action from policy
            s_idx = self.env.state_to_index(current_state)
            action = policy[s_idx]
            
            # take action
            next_state = self.env.get_next_state(current_state, action)
            reward = self.env.get_reward(current_state, action, next_state)
            
            # handle cliff
            if next_state in self.env.cliff_states:
                print(f"Step {step + 1}: Fell into cliff! Reset to start.")
                next_state = self.env.start_state
            
            path.append(next_state)
            total_reward += reward
            current_state = next_state
            
        print(f"Total reward: {total_reward}")
        return path, total_reward
   
   # visualize value function and policy
    def visualize_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # plot 1: value function heatmap
        ax1 = axes[0, 0]
        values_grid = self.V.reshape(self.env.height, self.env.width)
        im1 = ax1.imshow(values_grid, cmap='viridis')
        ax1.set_title('Value Function')
        plt.colorbar(im1, ax=ax1)
        
        # add text annotations
        for i in range(self.env.height):
            for j in range(self.env.width):
                ax1.text(j, i, f'{values_grid[i, j]:.1f}', 
                        ha='center', va='center', color='white', fontsize=8)
        
        # plot 2: environment layout
        ax2 = axes[0, 1] 
        layout = np.zeros((self.env.height, self.env.width))
        
        # color code: 0=normal, 1=start, 2=goal, 3=cliff
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                if state == self.env.start_state:
                    layout[i, j] = 1
                elif state == self.env.goal_state:
                    layout[i, j] = 2
                elif state in self.env.cliff_states:
                    layout[i, j] = 3
                else:
                    layout[i, j] = 0
        
        colors = ['lightgray', 'green', 'gold', 'red']
        im2 = ax2.imshow(layout, cmap='Set1')
        ax2.set_title('Environment Layout\n(Green=Start, Gold=Goal, Red=Cliff)')
        
        # plot 3: policy visualization
        ax3 = axes[1, 0]
        policy = self.extract_policy()
        action_symbols = ['↑', '→', '↓', '←']
        
        # create background
        ax3.imshow(layout, cmap='Set1', alpha=0.3)
        ax3.set_title('Optimal Policy')
        
        # add arrows
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                if state in [self.env.start_state, self.env.goal_state] or state in self.env.cliff_states:
                    continue
                    
                s_idx = self.env.state_to_index(state)
                action = policy[s_idx]
                ax3.text(j, i, action_symbols[action], ha='center', va='center',
                        fontsize=16, fontweight='bold')
        
        # plot 4: value convergence
        ax4 = axes[1, 1]
        if len(self.value_history) > 1:
            # Plot convergence for some interesting states
            start_idx = self.env.state_to_index(self.env.start_state)
            goal_idx = self.env.state_to_index(self.env.goal_state)
            middle_idx = self.env.state_to_index((1, 6))  # Middle of grid
            
            iterations = range(len(self.value_history))
            ax4.plot([V[start_idx] for V in self.value_history], 'g-o', label='Start', markersize=3)
            ax4.plot([V[goal_idx] for V in self.value_history], 'b-s', label='Goal', markersize=3)
            ax4.plot([V[middle_idx] for V in self.value_history], 'r-^', label='Middle', markersize=3)
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Value')
            ax4.set_title('Value Convergence')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # create cliff walking environment
    env = CliffWalkingEnvironment()
    env.print_environment()
    
    # run value iteration
    vi = ValueIterationCliff(env, gamma=0.9, theta=1e-6)
    
    print("\nInitial Value Function:")
    vi.print_value_function()
    
    # run the algorithm
    vi.run_value_iteration()
    
    print("\nFinal Results:")
    print("=" * 50)
    vi.print_value_function()
    vi.print_policy()
    
    # simulate an episode
    print("\nSimulating Optimal Policy:")
    print("-" * 30)
    path, reward = vi.simulate_episode()
    
    # visualize results
    vi.visualize_results()
    
    # print some insights
    print("\nInsights:")
    print("-" * 20)
    start_idx = env.state_to_index(env.start_state)
    print(f"Value of start state: {vi.V[start_idx]:.2f}")
    print(f"This represents the expected return from the start following the optimal policy")
    
    # check if the policy avoids the cliff
    policy = vi.extract_policy()
    cliff_adjacent = [(2, j) for j in range(1, 11)]  # States above cliff
    
    print(f"\nPolicy near cliff (row above cliff):")
    for state in cliff_adjacent:
        s_idx = env.state_to_index(state)
        action = policy[s_idx]
        action_name = env.action_names[action]
        print(f"State {state}: {action_name}")
