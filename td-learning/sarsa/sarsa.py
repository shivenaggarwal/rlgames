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


class SARSAAgent:
    # sarsa stands for state action reward state action
    # main diff from q learning is:
    # sarsa is on policy ie learns about the policy its acutally following
    # q learning is off policy ie learns about optimal policy regardess of whats its doing

    # SARSA update: Q(s,a) <- Q(s,a) + alpha[r + gamma * Q(s',a') - Q(s,a)]
    # where a' is the ACTUAL next action chosen by the policy

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
        self.lr = lr  # α (alpha) learning rate
        self.gamma = discount_factor  # γ (gamma) discount factor
        self.epsilon = epsilon  # ε (epsilon) exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        print("SARSA Agent created!")
        print(f"Learning rate (α): {self.lr}")
        print(f"Discount factor (γ): {self.gamma}")
        print(f"Initial exploration rate (ε): {self.epsilon}")

    def get_action(self, state, training=True):
        # choose action using ε-greedy policy
        if training and random.random() < self.epsilon:
            # exploration
            action = random.randint(0, self.n_actions - 1)
            print(f"  SARSA Exploring: random action {action}")
            return action
        else:
            # exploitation
            q_values = self.q_table[state]
            action = np.argmax(q_values)
            print(f"  SARSA Exploiting: best action {action} (Q-values: {q_values})")
            return action

    def update(self, state, action, reward, next_state, next_action, done):
        # SARSA: Q(s,a) <- Q(s,a) + alpha [r + gamma * Q(s',a') - Q(s,a)]
        # this here is the key diff between qlearn algo and SARSA
        # SARSA uses the ACTUAL next action (next_action),
        # not the maximum Q-value from next_state

        # get current Q-value
        current_q = self.q_table[state][action]

        if done:
            # terminal state no next action
            target_q = reward
            print(f"  SARSA Terminal: target = {reward}")
        else:
            # non terminal use Q-value of the actual next action
            next_q = self.q_table[next_state][next_action]
            target_q = reward + self.gamma * next_q
            print(
                f"  SARSA Non-terminal: target = {reward} + {self.gamma} * Q({
                    next_state
                },{next_action}) = {target_q}"
            )
            print(f"    Next action Q-value: {next_q}")

        # calculate TD error and update
        td_error = target_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[state][action] = new_q

        print(
            f"  SARSA Q-update: {current_q:.2f} → {new_q:.2f} (error: {td_error:.2f})"
        )

    def decay_epsilon(self):
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if old_epsilon != self.epsilon:
            print(f"Epsilon decayed: {old_epsilon:.3f} → {self.epsilon:.3f}")


def train_sarsa_step_by_step(episodes=5, max_steps_per_episode=50):
    # in Q-learning: we only need current action, then update using max Q(s',a')
    # in SARSA: we need BOTH current action AND next action for the update

    env = MazeEnv()
    agent = SARSAAgent(n_actions=env.n_actions)

    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):
        print(f"\n{'=' * 20} EPISODE {episode + 1} {'=' * 20}")

        # SARSA Step 1: initialize S
        state = env.reset()
        print(f"Starting at state: {state}")

        # SARSA Step 2: choose A from S using policy
        action = agent.get_action(state)
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        print(f"Initial action chosen: {action} ({action_names[action]})")

        total_reward = 0
        steps = 0

        # episode loop
        while steps < max_steps_per_episode:
            print(f"\nStep {steps + 1}:")
            print(f"Current state: {state}, Action: {action} ({action_names[action]})")

            # SARSA Step 3: take action A, observe R and S'
            next_state, reward, done = env.step(action)
            print(
                f"Environment response: next_state={next_state}, reward={reward}, done={
                    done
                }"
            )

            if done:
                # terminal state update and break
                print("Terminal state reached - updating Q-table")
                agent.update(state, action, reward, next_state, None, done)
                total_reward += reward
                steps += 1
                break

            # SARSA Step 4: choose A' from S' using policy
            next_action = agent.get_action(next_state)
            print(f"Next action chosen: {next_action} ({action_names[next_action]})")

            # SARSA Step 5: update Q(S,A) using actual next action A'
            print(f"Updating Q-table with SARSA rule:")
            agent.update(state, action, reward, next_state, next_action, done)

            # SARSA Step 6: S <- S', A <- A'
            state = next_state
            action = next_action

            total_reward += reward
            steps += 1

            print("-" * 40)

        print(f"\nEPISODE {episode + 1} COMPLETE")
        print(f"Total reward: {total_reward}")
        print(f"Steps taken: {steps}")

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # show sample Q-values
        print(f"\nSample Q-values:")
        sample_states = list(agent.q_table.keys())[:3]
        for sample_state in sample_states:
            print(f"  State {sample_state}: {agent.q_table[sample_state]}")

    return agent, episode_rewards, episode_steps


def compare_sarsa_vs_qlearning(episodes=200):
    print("=" * 60)
    print("COMPARING SARSA vs Q-LEARNING")
    print("=" * 60)

    env = MazeEnv()

    print("\n--- Training SARSA ---")
    sarsa_agent = SARSAAgent(n_actions=env.n_actions, epsilon=0.1)
    sarsa_rewards = []

    for episode in range(episodes):
        state = env.reset()
        action = sarsa_agent.get_action(state, training=True)
        total_reward = 0
        steps = 0

        while steps < 50:
            next_state, reward, done = env.step(action)

            if done:
                sarsa_agent.update(state, action, reward, next_state, None, done)
                total_reward += reward
                break

            next_action = sarsa_agent.get_action(next_state, training=True)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        sarsa_agent.decay_epsilon()
        sarsa_rewards.append(total_reward)

        if episode % 50 == 0:
            avg_reward = (
                np.mean(sarsa_rewards[-50:])
                if len(sarsa_rewards) >= 50
                else np.mean(sarsa_rewards)
            )
            print(f"SARSA Episode {episode}, Avg Reward: {avg_reward:.2f}")

    print("\n--- Training Q-learning ---")

    class SimpleQLearning:
        def __init__(
            self, n_actions, lr=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995
        ):
            self.n_actions = n_actions
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.q_table = defaultdict(lambda: np.zeros(n_actions))

        def get_action(self, state, training=True):
            if training and random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            return np.argmax(self.q_table[state])

        def update(self, state, action, reward, next_state, done):
            current_q = self.q_table[state][action]
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_state])
            self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

        def decay_epsilon(self):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    qlearning_agent = SimpleQLearning(n_actions=env.n_actions, epsilon=0.1)
    qlearning_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while steps < 50:
            action = qlearning_agent.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            qlearning_agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        qlearning_agent.decay_epsilon()
        qlearning_rewards.append(total_reward)

        if episode % 50 == 0:
            avg_reward = (
                np.mean(qlearning_rewards[-50:])
                if len(qlearning_rewards) >= 50
                else np.mean(qlearning_rewards)
            )
            print(f"Q-Learning Episode {episode}, Avg Reward: {avg_reward:.2f}")

    # plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sarsa_rewards, label="SARSA", alpha=0.7)
    plt.plot(qlearning_rewards, label="Q-Learning", alpha=0.7)
    plt.title("Learning Curves: SARSA vs Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # moving averages
    window = 20
    if len(sarsa_rewards) >= window:
        sarsa_avg = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")
        qlearn_avg = np.convolve(
            qlearning_rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            range(window - 1, len(sarsa_rewards)),
            sarsa_avg,
            label="SARSA (smoothed)",
            linewidth=2,
        )
        plt.plot(
            range(window - 1, len(qlearning_rewards)),
            qlearn_avg,
            label="Q-Learning (smoothed)",
            linewidth=2,
        )
    plt.title("Smoothed Learning Curves")
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return sarsa_agent, qlearning_agent


if __name__ == "__main__":
    # SARSA trainig
    print("\n" + "=" * 60)
    print("STEP-BY-STEP SARSA TRAINING")
    print("=" * 60)

    agent, rewards, steps = train_sarsa_step_by_step(episodes=3)

    # compare SARSA vs Qlearning
    print("\n" + "=" * 60)
    print("COMPARING SARSA vs Q-LEARNING")
    print("=" * 60)

    sarsa_agent, qlearning_agent = compare_sarsa_vs_qlearning(episodes=200)
