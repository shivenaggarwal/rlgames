import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

# neural network for value function approx


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[64, 64], activation="relu"):
        super(ValueNetwork, self).__init__()

        # choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # built neural network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # initialize layers
        self._init_weights()

    # initialize weight with xavier initialization

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    # forward pass thru the network

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # ensure proper shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        return self.network(state).squeeze(-1)


# experience replay to store transitions
class ExperienceReplay:
    def __init__(self, capacity=10000):
        # automatically removes oldest items when full
        self.buffer = deque(maxlen=capacity)

    # adds a transition to the buffer
    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    # sample a batch of transitions
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # unpacks the list of tuples into a sepetate list for each component
        states, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


# td learninig with non linear function approx
class NonLinearTD:
    def __init__(
        self,
        state_dim,
        lr=0.001,
        gamma=0.99,
        hidden_dims=[64, 64],
        target_update_freq=100,
    ):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0

        # main value network
        self.value_net = ValueNetwork(state_dim, hidden_dims)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # target network stability
        # creates an identical target network
        self.target_net = ValueNetwork(state_dim, hidden_dims)
        # copies weights from main network to target netwrork
        self.target_net.load_state_dict(self.value_net.state_dict())

        # experince replay
        self.replay_buffer = ExperienceReplay()

        # tracking
        self.losses = []  # init empty list to track loss

    # get value estimate for a state
    def get_value(self, state):
        with torch.no_grad():
            # passes thru network and extracts scalar value
            return self.value_net(state).item()

    # update value functuon with td error
    def update(self, state, reward, next_state, done, batch_size=32):
        # store experince
        self.replay_buffer.push(state, reward, next_state, done)

        # need enough replays to sample from
        if len(self.replay_buffer) < batch_size:
            return

        # sample batch from replay buffer
        states, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # convert to tensors
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # current value estimates
        current_values = self.value_net(states)

        # target value estimates (using target network)
        with torch.no_grad():
            next_values = self.target_net(next_states)
            # computes td target: r + gamma * V(s') if not done otherwise r
            targets = rewards + self.gamma * next_values * (1 - dones)

        # compute td error and losss
        td_error = targets - current_values
        loss = F.mse_loss(current_values, targets)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping for stability
        # clips gradients to max norm of 1.0 to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

        self.optimizer.step()

        # update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

        # track loss
        self.losses.append(loss.item())

        return loss.item()

    # saves the current model
    def save_model(self, filepath):
        torch.save(
            {
                "value_net_state_dict": self.value_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    # load the saved model
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



# monte carlo learning with non linear value function approx 
class MonteCarloVFA:
    def __init__(self, state_dim, lr=0.001, hidden_dims=[64, 64]):
        self.value_net = ValueNetwork(state_dim, hidden_dims)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.losses = []

    # get value estimate of a state 
    def get_value(self, state):
        with torch.no_grad():
            return self.value_net(state).item()

        
    # update value function using montecarlo returns from full episode 
    def update_episode(self, episode_states, episode_rewards, gamma=0.99):
        # calculate returns for each state in the episode 
        returns = [] 
        G = 0 # returns accumulator
        for reward in reversed(episode_rewards):
            G = gamma * G + reward # computes discounted return: G_t = r_t * gamma * G_t+1
            returns.insert(0, G)

        # convert to tensors
        states = torch.FloatTensor(np.array(episode_states))
        targets = torch.FloatTensor(returns)

        # get current value estimates 
        predicted_values = self.value_net(states).squeeze()

        # compute loss 
        loss = F.mse_loss(predicted_values, targets)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()


# semi gradient TD(0) with non linear function approximation
class SemiGradientTD:
    def __init__(self, state_dim, lr=0.001, gamma=0.99, hidden_dims=[64, 64]):
        self.gamma = gamma
        self.value_net = ValueNetwork(state_dim, hidden_dims)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.losses = []
    
    # get value estimate for a state
    def get_value(self, state):
        with torch.no_grad():
            return self.value_net(state).item()

    # single step TD update (semi-gradient)
    def update_step(self, state, reward, next_state, done):
         # current value estimate
        current_value = self.value_net(state)
        
        # TD target (no gradient through next state value)
        with torch.no_grad():
            if done:
                td_target = reward
            else:
                next_value = self.value_net(next_state).item()
                td_target = reward + self.gamma * next_value
        
        td_target = torch.FloatTensor([td_target])
        
        # compute loss
        loss = F.mse_loss(current_value, td_target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()

# compare different non-linear VFA methods 
def test_value_function_methods():
    # create a simple 2D state space
    state_dim = 2
    
    # initialize different VFA methods
    td_agent = NonLinearTD(state_dim, lr=0.001, gamma=0.99)
    semi_grad_agent = SemiGradientTD(state_dim, lr=0.001, gamma=0.99)
    mc_agent = MonteCarloVFA(state_dim, lr=0.001)
    
    print("Comparing different non-linear VFA methods...")
    
    # train TD agent with experience replay
    print("\n1. Training TD with Experience Replay...")
    for episode in range(500):
        state = np.random.uniform(-1, 1, 2)
        
        # simple reward function
        reward = -np.sum(state**2)  # reward peaks at origin
        done = np.random.random() < 0.1  # random termination
        next_state = state + np.random.normal(0, 0.1, 2)
        
        loss = td_agent.update(state, reward, next_state, done)
        
        if episode % 100 == 0 and loss is not None:
            print(f"Episode {episode}, TD Loss: {loss:.4f}")
    
    # train semi gradient TD agent
    print("\n2. Training Semi-Gradient TD...")
    for episode in range(500):
        state = np.random.uniform(-1, 1, 2)
        reward = -np.sum(state**2)
        done = np.random.random() < 0.1
        next_state = state + np.random.normal(0, 0.1, 2)
        
        loss = semi_grad_agent.update_step(state, reward, next_state, done)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Semi-Grad TD Loss: {loss:.4f}")
    
    # train Monte Carlo agent
    print("\n3. Training Monte Carlo...")
    for episode in range(200):  # fewer episodes since we need complete episodes
        # generate a complete episode
        episode_states = []
        episode_rewards = []
        
        state = np.random.uniform(-1, 1, 2)
        for step in range(10):  # 10-step episodes
            episode_states.append(state.copy())
            reward = -np.sum(state**2)
            episode_rewards.append(reward)
            
            # simple dynamics
            state += np.random.normal(0, 0.1, 2)
            state = np.clip(state, -1, 1)  # keep in bounds
        
        loss = mc_agent.update_episode(episode_states, episode_rewards)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, MC Loss: {loss:.4f}")
    
    # test all methods on same states
    print("\n4. Comparing learned value functions:")
    test_states = [
        np.array([0.0, 0.0]),    # origin (should have high value)
        np.array([0.5, 0.5]),    # medium distance
        np.array([1.0, 1.0]),    # far from origin (low value)
        np.array([-0.5, 0.5]),   # mixed coordinates
    ]
    
    print("State\t\tTD+Replay\tSemi-Grad TD\tMonte Carlo")
    print("-" * 60)
    for state in test_states:
        td_val = td_agent.get_value(state)
        semi_val = semi_grad_agent.get_value(state)
        mc_val = mc_agent.get_value(state)
        print(f"{state}\t{td_val:.3f}\t\t{semi_val:.3f}\t\t{mc_val:.3f}")
    
    return td_agent, semi_grad_agent, mc_agent


# demonstrate key properties of non-linear function approximation
def test_function_approximation_properties():
    print("\n" + "="*50)
    print("Testing Function Approximation Properties")
    print("="*50)
    
    state_dim = 1  # 1D for easy visualization
    agent = NonLinearTD(state_dim, lr=0.001, gamma=0.95)
    
    # create training data with known value function
    # true value function: V(s) = s^2 for s in [-1, 1]
    training_data = []
    for _ in range(1000):
        state = np.random.uniform(-1, 1, 1)
        true_value = state[0]**2
        
        # add noise to simulate environment
        noisy_reward = true_value + np.random.normal(0, 0.1)
        next_state = np.array([np.clip(state[0] + np.random.normal(0, 0.1), -1, 1)])
        done = np.random.random() < 0.05
        
        training_data.append((state, noisy_reward, next_state, done))
    
    # train the agent
    print("Training on quadratic value function...")
    for i, (state, reward, next_state, done) in enumerate(training_data):
        loss = agent.update(state, reward, next_state, done)
        if i % 200 == 0 and loss is not None:
            print(f"Step {i}, Loss: {loss:.4f}")
    
    # test approximation quality
    print("\nTesting approximation quality:")
    print("State\tTrue Value\tApprox Value\tError")
    print("-" * 40)
    
    total_error = 0
    test_points = np.linspace(-1, 1, 11)
    for s in test_points:
        state = np.array([s])
        true_val = s**2
        approx_val = agent.get_value(state)
        error = abs(true_val - approx_val)
        total_error += error
        
        print(f"{s:.1f}\t{true_val:.3f}\t\t{approx_val:.3f}\t\t{error:.3f}")
    
    avg_error = total_error / len(test_points)
    print(f"\nAverage approximation error: {avg_error:.3f}")
    
    return agent


if __name__ == "__main__":
    # run tests
    print("Non-Linear Value Function Approximation Implementation")
    print("=" * 55)
    
    # test different methods
    agents = test_value_function_methods()
    
    # test approximation properties
    test_agent = test_function_approximation_properties()

