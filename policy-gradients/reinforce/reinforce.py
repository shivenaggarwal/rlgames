from functools import total_ordering
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

# yoo this is reinforce algo

class PolicyNetwork(nn.Module):
    """simple policy network that ouputs action probs"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # softmax converts raw logits to probs that sum to 1 which we need for valid policu pie(a|s)
        return F.softmax(x, dim=1)


class REINFORCEAgent:
    """implementing the policy gradient theorem"""

    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # initalize policy network and optimizer
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # storage for episode data
        # store log probs log pie(a_t|s_t) for each time step
        self.log_probs = []
        self.rewards = []  # stores reward r_t

        # log probs because divide the gradient with the policy which is basically gradient of the log. we do this for efficiency.EOFError

    def select_action(self, state):
        """select an action using curr policy"""
        state = torch.FloatTensor(state).unsqueeze(
            0
        )  # convert to tensor and add batch dim
        probs = self.policy_net(state)
        # Categorical distribution for sampling discrete actions
        m = Categorical(probs)
        action = m.sample()

        # store log probs for policy gradient update
        self.log_probs.append(m.log_prob(action))

        return action.item()  # return as python integer

        # here we are basically sampling from our policy rather than taking the most probable action

    def store_reward(self, reward):
        """store reward for current time step"""
        self.rewards.append(reward)

    def calculate_returns(self):
        """
        calculate discounted returns (G_t) for each time step
        this is the monte carlo estimate of state value
        """
        returns = []
        G = 0

        # work backwards from end of the episode
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G  # bellman equation
            returns.insert(0, G)

        return returns

    def update_policy(self):
        """
        update policy using reinforce algo
        loss = -grad_theta J(theta) = -E[grad_theta log pie(a | s) * G_t]
        """
        returns = self.calculate_returns()

        # convert to tensor and normalize
        returns = torch.FloatTensor(returns)
        # not required just doing this for better results
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # this basically reduces the variance w/o changing the expected grad

        policy_loss = []  # init loss accumulator

        # calculate policy grdient loss
        for log_prob, G in zip(self.log_probs, returns):  # iterate thru timesteps
            # negative because we want to maximize expected return
            policy_loss.append(-log_prob * G)

        # update policy network
        self.optimizer.zero_grad()
        # combine all timestep losses
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # clear episode data
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()


def train_reinforce(env_name="CartPole-v1", num_episodes=1000):
    """train reinforce agent on given env"""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)
    
    scores = []
    losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        # run episode 
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.store_reward(reward)
            total_reward += reward

            if terminated or truncated:
                break

            state = next_state

        # update policy at the end of the episode
        loss = agent.update_policy()

        scores.append(total_reward)
        losses.append(loss)

        # print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
        
        # early stopping if solved
        if np.mean(scores[-100:]) >= 195.0:
            print(f"Environment solved in {episode} episodes!")
            break

    env.close()
    return agent, scores, losses

# little notes for myself
# reinforce is a monte carlo method, it needs complete episode returns G_t before updating

def plot_results(scores, losses):
    """plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(scores)
    ax1.set_title('Scores over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    
    ax2.plot(losses)
    ax2.set_title('Policy Loss over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# testing
if __name__ == "__main__":
    print("Training REINFORCE agent...")
    agent, scores, losses = train_reinforce()
    
    print("\nPlotting results...")
    plot_results(scores, losses)
    
    # test trained agent
    print("\nTesting trained agent...")
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    
    total_reward = 0
    while True:
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Test episode reward: {total_reward}")
            break
    
    env.close()


# what i noticed after running the algo 
# early episodes: lucky trajectory -> high return -> large gradient update -> good policy
# later episodes: unlucky trajectory -> low return -> large gradient update in wrong direction -> bad policy
# the agent "forgets" what it learned due to noisy gradient estimates
# ways to fix could be adding gradient clipping or reducing the lr or increasing episode count for averaging
