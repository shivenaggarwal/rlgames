import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class MazeEnv:
    def __init__(self):
        # 0=empty space, 1=wall, 2=goal, 3=start pos

        self.maze = np.array(
            [
                [3, 0, 1, 0, 0],  # row 0: start at (0,0), wall at (0,2)
                [0, 0, 1, 0, 1],  # row 1: walls at (1,2) and (1,4)
                [0, 0, 0, 0, 1],  # row 2: wall at (2,4)
                [1, 1, 0, 1, 1],  # row 3: only (3,2) is free
                [0, 0, 0, 2, 0],  # row 4: goal at (4,3)
            ]
        )

        # define start and goal pos
        self.start_pos = (0, 0)
        self.goal_pos = (4, 3)

        # current pos of agent
        self.current_pos = self.start_pos

        # define possible actions as (row_change, col_change)
        # 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.n_actions = len(self.actions)

        print("Env Created")
        print(f"Start position: {self.start_pos}")
        print(f"Goal position: {self.goal_pos}")
        print(f"Number of actions: {self.n_actions}")

    def reset(self):
        # reset the env to initial state. this is called start of a new episode
        self.current_pos = self.start_pos
        return self.current_pos

    def is_valid_position(self, pos):
        # check if a positon is valid(within bounds and not a wall)
        row, col = pos

        # check if the positon is within the maze boundary
        if row < 0 or row >= self.maze.shape[0]:
            return False
        if col < 0 or col >= self.maze.shape[1]:
            return False

        # check if position is not a wall(maze value not equals to 1)
        return self.maze[row, col] != 1

    def step(self, action):
        # take an action in the env

        # calculate where we will end up if we take the action
        dr, dc = self.actions[action]  # get row and column change
        new_pos = (self.current_pos[0] + dr, self.current_pos[1] + dc)

        # check if new pos is valid
        if self.is_valid_position(new_pos):
            # move is valid update position
            self.current_pos = new_pos

        else:
            # move is invalid no state change
            pass

        # calculate reward based on where we end up
        if self.current_pos == self.goal_pos:
            reward = 100
            done = True
            print(f"Goal reached at {self.current_pos}")

        else:
            reward = -1
            done = False

        return self.current_pos, reward, done

    def render(self, q_table=None):
        # visualize the maze and optionally show policy
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # create display maze
        display_maze = self.maze.copy().astype(float)

        # color coding: walls=black, free=white, goal=green, current=red
        colors = np.zeros((*self.maze.shape, 3))

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:  # wall
                    colors[i, j] = [0, 0, 0]  # black
                elif (i, j) == self.goal_pos:  # goal
                    colors[i, j] = [0, 1, 0]  # green
                elif (i, j) == self.current_pos:  # current position
                    colors[i, j] = [1, 0, 0]  # red
                else:  # Free space
                    colors[i, j] = [1, 1, 1]  # white

        ax.imshow(colors)

        # add policy arrows if Q-table provided
        if q_table is not None:
            arrow_symbols = ["↑", "→", "↓", "←"]
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    # not wall or goal
                    if self.maze[i, j] != 1 and (i, j) != self.goal_pos:
                        state = (i, j)
                        if state in q_table:
                            best_action = np.argmax(q_table[state])
                            ax.text(
                                j,
                                i,
                                arrow_symbols[best_action],
                                ha="center",
                                va="center",
                                fontsize=20,
                                fontweight="bold",
                            )

        ax.set_title("Maze Environment")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()


# Q-learning learns a "Q-table" that tells us the quality (Q-value) of taking
# each action in each state. the Q-value represents expected future reward


class QLearningAgent:
    def __init__(
        self,
        n_actions,
        lr=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.n_actions = n_actions
        self.lr = lr  # alpha
        self.gamma = discount_factor  # gamma
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table maps (State, action) -> Q-value
        # using default dict so new sates automatically gets zero Q-value
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        print("Q-Learning Agent created!")
        print(f"Learning rate (α): {self.lr}")
        print(f"Discount factor (γ): {self.gamma}")
        print(f"Initial exploration rate (ε): {self.epsilon}")

    def get_action(self, state, training=True):
        # chooose an action according to epsilon greedy policy. gives balance between exploration and exploitation
        if training and random.random() < self.epsilon:
            # exploration:: choose random action
            action = random.randint(0, self.n_actions - 1)
            print(f"  Exploring: random action {action}")
            return action

        else:
            # exploitation: choose best action according to Q-table
            # this uses what we have learned so far
            q_values = self.q_table[state]
            action = np.argmax(q_values)  # action with the highest q-value
            print(f"  Exploiting: best action {action} (Q-values: {q_values})")
            return action

    def update(self, state, action, reward, next_state, done):
        # update Q-table using QLearning update rule
        # Q(s,a) <- Q(s,a) + alpha[r + gamma * max[Q(s', a')] - Q(s,a)]

        # get current Q-value for this state action pair
        current_q = self.q_table[state][action]

        if done:
            # terminal state no future rewards possible
            target_q = reward
            print(f"  Terminal state: target = {reward}")

        else:
            # non terminal state consider future rewards
            # max Q(s',a') = best possible Q-value from next state
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
            print(
                f"  Non-terminal: target = {reward} + {self.gamma} * {max_next_q} = {
                    target_q
                }"
            )

        # calculate the error how wrong our current q value is
        td_error = target_q - current_q

        # update q-value move towards target by lr amt
        new_q = current_q + self.lr * td_error
        self.q_table[state][action] = new_q

        print(f"  Q-update: {current_q:.2f} → {new_q:.2f} (error: {td_error:.2f})")

    def decay_epsilon(self):
        # reduce exploration over time
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if old_epsilon != self.epsilon:
            print(f"Epsilon decayed: {old_epsilon:.3f} → {self.epsilon:.3f}")


# training loop


def train_q_learning(episodes=1000, render_every=100):
    env = MazeEnv()
    agent = QLearningAgent(n_actions=env.n_actions)

    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:  # max steps to prevent infinite loops
            # choose action
            action = agent.get_action(state)

            # take step in env
            next_state, reward, done = env.step(action)

            # update Q-table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        # decay epsilon
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # print progress
        if episode % render_every == 0:
            avg_reward = (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) >= 100
                else np.mean(episode_rewards)
            )
            avg_steps = (
                np.mean(episode_steps[-100:])
                if len(episode_steps) >= 100
                else np.mean(episode_steps)
            )
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {
                    avg_steps:.2f}, Epsilon: {agent.epsilon:.3f}"
            )

    return agent, episode_rewards, episode_steps


def test_agent(agent, episodes=10):
    # testing traine agent
    env = MazeEnv()
    success_count = 0
    total_steps = []

    for episode in range(episodes):
        state = env.reset()
        steps = 0
        done = False

        print(f"\n--- Test Episode {episode + 1} ---")
        path = [state]

        while not done and steps < 50:
            action = agent.get_action(state, training=False)  # no exploration
            next_state, reward, done = env.step(action)
            path.append(next_state)
            state = next_state
            steps += 1

        if done:
            success_count += 1
            print(f"Success! Reached goal in {steps} steps")
            print(f"Path: {path}")
        else:
            print(f"Failed to reach goal in {steps} steps")

        total_steps.append(steps)

    print(f"\nTest Results:")
    print(
        f"Success Rate: {success_count}/{episodes} ({
            100 * success_count / episodes:.1f}%)"
    )
    if success_count > 0:
        successful_steps = [
            steps for i, steps in enumerate(total_steps) if i < success_count
        ]
        print(f"Average steps to goal: {np.mean(successful_steps):.2f}")


def plot_training_results(episode_rewards, episode_steps):
    # plot training progres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # plot rewards
    ax1.plot(episode_rewards, alpha=0.6)
    # moving average
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode="valid")
        ax1.plot(range(99, len(episode_rewards)), moving_avg, "r-", linewidth=2)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True)

    # plot steps
    ax2.plot(episode_steps, alpha=0.6)
    # moving average
    if len(episode_steps) >= 100:
        moving_avg = np.convolve(episode_steps, np.ones(100) / 100, mode="valid")
        ax2.plot(range(99, len(episode_steps)), moving_avg, "r-", linewidth=2)
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # train the agent
    print("Training Q-learning agent...")
    agent, rewards, steps = train_q_learning(episodes=1000, render_every=200)

    # plot training results
    plot_training_results(rewards, steps)

    # visualize final policy
    env = MazeEnv()
    print("\nFinal learned policy:")
    env.render(agent.q_table)

    # test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, episodes=10)

    # print some Q-values for inspection
    print("\nSample Q-values:")
    for state in list(agent.q_table.keys())[:5]:
        print(f"State {state}: {agent.q_table[state]}")
