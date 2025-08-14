import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

# tile coding for a continous state spaces
# creates multiple overlapping tilings to approximate value functions
# its like covering a mountain car 2d space (pos x vel) with a grid
# nowimagine 8 such grids, each slightly offset from the others
# for any state, you're in exactly one tile from each grid, these are our "active features"


class TileCoder:
    def __init__(self, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        self.num_tilings = num_tilings  # overlapping grids
        self.tiles_per_dim = tiles_per_dim  # 8x8 = 64 tiles per grid
        # mountain car limits position from -1.2 to 0.6, velocity from -0.07 to 0.07
        self.state_bounds = state_bounds or [(-1.2, 0.6), (-0.07, 0.07)]

        # calculate tile size for each dim
        self.tile_sizes = []
        for low, high in self.state_bounds:
            # FIXED: Added division
            self.tile_sizes.append((high - low) / tiles_per_dim)

        # position: (0.6 - (-1.2)) / 8 = 1.8 / 8 = 0.225 units wide
        # velocity: (0.07 - (-0.07)) / 8 = 0.14 / 8 = 0.0175 units wide

        # calculate offset for each tiling - FIXED: Moved outside the first loop
        self.offsets = []
        for i in range(num_tilings):
            offset = []
            for dim in range(len(self.state_bounds)):
                offset.append(i * self.tile_sizes[dim] / num_tilings)
            self.offsets.append(offset)

        # grid 0: offset = [0.0, 0.0] (no offset)
        # grid 1: offset = [0.225/8, 0.0175/8] = [0.028, 0.002]
        # grid 2: offset = [2×0.028, 2×0.002] = [0.056, 0.004], etc
        # this ensures 8 grids are evenly distrubuted offsets of each others

    # returns the active tiles for a given state.
    # each tiling contributes one active tile.
    def get_tiles(self, state):
        tiles = []
        for tiling_idx in range(self.num_tilings):
            tile_coords = []
            for dim in range(len(state)):  # loop thru 2 dim
                # for each grid and each dim (pos, vel)
                # adjust state by tiling offset
                adjusted_state = state[dim] - self.offsets[tiling_idx][dim]

                # original position: -0.5
                # grid 1 offset: 0.028
                # adjusted position: -0.5 - 0.028 = -0.528

                # find which tile this falls into
                low, _ = self.state_bounds[dim]
                tile_coord = int((adjusted_state - low) / self.tile_sizes[dim])

                # for position: int((-0.528 - (-1.2)) / 0.225) = int(0.672 / 0.225) = int(2.99) = 2
                # so we are in positon 2

                # clamp to valid range ensure the coordinate ranges between 0 to 7
                tile_coord = max(0, min(tile_coord, self.tiles_per_dim - 1))
                tile_coords.append(tile_coord)

            # convert coordinates to unique tile index
            tile_index = 0
            multiplier = 1
            for coord in reversed(tile_coords):
                tile_index += coord * multiplier
                multiplier *= self.tiles_per_dim

            # this is basically converting 2d coords (row, col) to a single index
            # tile_index = 5 * 1 + 2 * 8 = 5 + 16 = 21

            # add tiling offset to make tiles unique across tilings
            tile_index += tiling_idx * (self.tiles_per_dim ** len(state))
            tiles.append(tile_index)

        return tiles


# linear func approximator using tile coding features
# Q(s,a) = sum(w_i * x_i) where x_i are tile coding features
class LinearFunctionApproximator:
    def __init__(self, num_actions, tile_coder, alpha=0.1):
        self.num_actions = num_actions  # 3 for mountain Car
        self.tile_coder = tile_coder
        self.alpha = alpha  # learning rate

        # weight vector using defaultdict for sparse representation
        # instead of storing weights for all possible states we only store weight for tiles we have actally seen and returns 0.0 for any unseen states
        self.weights = defaultdict(float)

    # getting the feature vector for each state action pair
    def get_features(self, state, action):
        tiles = self.tile_coder.get_tiles(state)

        # create unique features for each action by offsetting tile indices
        action_offset = action * 10000  # large offset to separate actions
        # this ensures Q(s,a) for different actions uses completely different weights

        features = [tile + action_offset for tile in tiles]
        return features

    # compute Q(s,a) = sum of weights for active get_features
    # Q(s,a) = w21 + w85 + w149 + ... = sum(w_i)
    # since these features are binary and only 8 features are active q value is just the sum of weights
    def q_value(self, state, action):
        features = self.get_features(state, action)
        return sum(self.weights[f] for f in features)

    # update weights using gradient descent: w += alpha * (target - Q(s,a)) * features
    def update(self, state, action, target):
        features = self.get_features(state, action)
        current_q = sum(self.weights[f] for f in features)
        error = target - current_q

        # update each active feature weight
        for f in features:
            self.weights[f] += self.alpha * error

    # get action with highest q value (greedy policy)
    def get_best_action(self, state):
        q_values = [self.q_value(state, a) for a in range(self.num_actions)]
        return np.argmax(q_values)

    # get q values for all action
    def get_action_values(self, state):
        return [self.q_value(state, a) for a in range(self.num_actions)]


# epsilon greedy policy for each action


def epsilon_greedy_policy(q_func, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(q_func.num_actions)  # random action
    else:
        return q_func.get_best_action(state)  # greedy action


# SARSA with linear function approximation using tile coding
def sarsa_with_function_approximation(
    env, num_episodes=500, alpha=0.1, epsilon=0.1, gamma=1.0
):
    # initialize tile coder and function approximator
    tile_coder = TileCoder(num_tilings=8, tiles_per_dim=8)
    q_func = LinearFunctionApproximator(env.action_space.n, tile_coder, alpha=alpha)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        # choose initial action
        action = epsilon_greedy_policy(q_func, state, epsilon)

        total_reward = 0
        steps = 0

        while True:
            # take action
            next_state, reward, done, truncated, _ = env.step(
                action
            )  # FIXED: Changed 'end' to 'env'
            total_reward += reward
            steps += 1

            if done or truncated or steps > 1000:
                # terminal state update
                target = reward
                q_func.update(state, action, target)
                break

            else:
                # choose next action
                next_action = epsilon_greedy_policy(q_func, next_state, epsilon)

                # SARSA update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
                target = reward + gamma * q_func.q_value(next_state, next_action)
                q_func.update(state, action, target)

                # move to next state and action
                state = next_state
                action = next_action

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(
                f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Length = {
                    avg_length:.2f}"
            )

    return q_func, episode_rewards, episode_lengths


# visualize the learned value function and policy
def visualize_learned_policy(q_func, env):
    # create a grid of states
    pos_range = np.linspace(-1.2, 0.6, 50)
    vel_range = np.linspace(-0.07, 0.07, 50)

    values = np.zeros((len(pos_range), len(vel_range)))
    policy = np.zeros((len(pos_range), len(vel_range)))

    for i, pos in enumerate(pos_range):
        for j, vel in enumerate(vel_range):
            state = np.array([pos, vel])

            # Get value (max Q-value over actions)
            q_values = q_func.get_action_values(state)
            values[i, j] = max(q_values)
            policy[i, j] = np.argmax(q_values)

    # plot value function
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(values, extent=[-0.07, 0.07, -1.2, 0.6], aspect="auto", origin="lower")
    plt.colorbar(label="Value")
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title("Learned Value Function")

    plt.subplot(1, 3, 2)
    plt.imshow(policy, extent=[-0.07, 0.07, -1.2, 0.6], aspect="auto", origin="lower")
    plt.colorbar(label="Action (0=Left, 1=Nothing, 2=Right)")
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title("Learned Policy")

    return plt


# test the learned policy


def test_learned_policy(q_func, env, num_test_episodes=5):
    print("\nTesting learned policy...")
    test_rewards = []

    for episode in range(num_test_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        steps = 0

        while steps < 1000:
            action = q_func.get_best_action(state)  # Greedy policy
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if done or truncated:
                break

        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    return test_rewards


if __name__ == "__main__":
    # create environment
    env = gym.make("MountainCar-v0")

    print("Training SARSA with Linear Function Approximation on Mountain Car...")
    print(
        f"State space: Position [{env.observation_space.low[0]:.2f}, {env.observation_space.high[0]:.2f}], "
        f"Velocity [{env.observation_space.low[1]:.2f}, {env.observation_space.high[1]:.2f}]"
    )
    print(f"Action space: {env.action_space.n} actions (0=Left, 1=Nothing, 2=Right)")

    # train the agent
    q_func, episode_rewards, episode_lengths = sarsa_with_function_approximation(
        env, num_episodes=500, alpha=0.1, epsilon=0.1, gamma=1.0
    )

    # plot training progress
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    # smooth the rewards for better visualization
    window_size = 20
    smoothed_rewards = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(range(window_size - 1, len(episode_rewards)), smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.title("Training Progress: Episode Rewards")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    smoothed_lengths = np.convolve(
        episode_lengths, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(range(window_size - 1, len(episode_lengths)), smoothed_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (smoothed)")
    plt.title("Training Progress: Episode Lengths")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # visualize learned policy
    viz_plt = visualize_learned_policy(q_func, env)
    plt.tight_layout()
    plt.show()

    # test the learned policy
    test_rewards = test_learned_policy(q_func, env)

    env.close()

    print(f"\nTraining completed!")
    print(
        f"Final average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}"
    )
    print(f"Number of weight parameters learned: {len(q_func.weights)}")
