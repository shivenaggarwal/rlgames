import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import (
    Categorical,
    Normal,
    # for continuous control tasks we model the policy as a multivariate gaussian pie(a|s) = N(mu(s), sigma(s))
)
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """policy network for discrete action"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class ValueNetwork(nn.Module):
    """value network for baseline(actor-critic)"""

    def __init__(
        self, state_size, hidden_size=128
    ):  # no action_size beause value func v(s) ouptuts a single scalar
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # no softmax or sigmoid because value can be any real number
        return self.fc3(x)


class ContinuousPolicyNetwork(nn.Module):
    """policy network for continuous actions (outpus mean and std)"""

    # action size now represents the dimentionality of continuous action vector
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # output layer for mean mu of gaussian distribution
        self.mean = nn.Linear(hidden_size, action_size)
        # output layer for log(std) of gaussian
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp for numerical stability.
        std = torch.exp(log_std.clamp(-20, 2))
        return mean, std

        # why seperate mean and std_log outputs?
        # - gaussian policy = pie(a|s) = N(mu(s), sigma(s))
        # - neural network outputs mu(s) and log(sigma(s))
        # - we use log(sigma) because sigma must be positive and exp of log(sigma) = sigma ensures this

        # clamp between -20 and 2 because
        # - lower bound: exp(-20) = 2 x 10 ^ -9 (very smalll exploration)
        # - upper bound: exp(2) = 7.4 (resonable exploration)
        # clamp prevents numerical instability (sigma -> 0 or sigma -> infinity)


class REINFORCEWithBaseline:
    """
    REINFORCE with baseline (actor-critic style)
    reduces variance by subtracting state value as baseline
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):  
        self.gamma = gamma

        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.log_probs = []
        self.values = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        # get action probabilities and state value
        probs = self.policy_net(state)
        value = self.value_net(state)

        # add numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=1, keepdim=True)

        m = Categorical(probs)
        action = m.sample()

        self.log_probs.append(m.log_prob(action))
        self.values.append(value)

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        # check for empty episode
        if len(self.rewards) == 0:
            return 0.0, 0.0
            
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        values = torch.cat(self.values).squeeze()

        # normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # calculate advantage (G_t - V(s_t))
        advantages = returns - values.detach()  # detach to prevent backprop through value

        # policy loss: -log pie(a|s) * advantage
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            # detach to prevent backprop through value
            policy_loss.append(-log_prob * advantage)

        # value loss: MSE between returns and predicted values
        value_loss = F.mse_loss(values, returns.detach())  # detach returns

        # update both networks
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean()  # average instead of sum
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)  # gradient clipping
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)  # gradient clipping
        self.value_optimizer.step()

        # clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []

        return policy_loss.item(), value_loss.item()


class ContinuousREINFORCE:
    """
    REINFORCE for continuous action spaces
    uses gaussian policy parameterized by neural network
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):  # lower LR
        self.gamma = gamma

        self.policy_net = ContinuousPolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        mean, std = self.policy_net(state)
        
        # add numerical stability
        std = torch.clamp(std, min=1e-6)
        
        dist = Normal(mean, std)
        action = dist.sample()

        # sum over action dimensions
        self.log_probs.append(dist.log_prob(action).sum())

        return action.squeeze().detach().numpy()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        # check for empty episode
        if len(self.rewards) == 0:
            return 0.0
            
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # stable normalization

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean()  # average instead of sum
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)  # gradient clipping
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

        return policy_loss.item()


class NaturalPolicyGradient:
    """
    natural policy gradient using fisher information matrix
    more principled update direction than vanilla policy gradient
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, damping=0.01):  # lower LR and damping
        self.gamma = gamma
        self.damping = damping
        self.step_size = lr  # store step size

        self.policy_net = PolicyNetwork(state_size, action_size)
        # NPG does manual parameter updates

        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        
        # add numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        m = Categorical(probs)
        action = m.sample()

        self.states.append(state)
        self.actions.append(action.item())

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def get_flat_params(self):
        """Get flattened parameters"""
        params = []
        for param in self.policy_net.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        """Set parameters from flattened vector"""
        prev_ind = 0
        for param in self.policy_net.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
            )
            prev_ind += flat_size

    def compute_fisher_vector_product(self, vector):
        """compute fisher information matrix times vector"""
        states = torch.FloatTensor(np.array(self.states))
        
        # Get current policy probabilities
        probs = self.policy_net(states)
        # add numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Sample from current policy for Fisher computation
        m = Categorical(probs)
        sampled_actions = m.sample()
        
        # Compute log probabilities
        log_probs = m.log_prob(sampled_actions)

        # compute gradients
        grads = torch.autograd.grad(
            log_probs.sum(), self.policy_net.parameters(), create_graph=True, retain_graph=True
        )

        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # fisher vector product
        grad_vector_product = torch.sum(flat_grad * vector)
        
        fisher_vector_product = torch.autograd.grad(
            grad_vector_product, self.policy_net.parameters(), retain_graph=True
        )
        fisher_vector_product = torch.cat(
            [grad.contiguous().view(-1) for grad in fisher_vector_product]
        )

        return fisher_vector_product + self.damping * vector

    def conjugate_gradient(self, b, nsteps=10, tol=1e-10): 
        """solve Ax = b using conjugate gradient - IMPROVED"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            # early stopping
            if rdotr < tol:
                break
                
            Ap = self.compute_fisher_vector_product(p)
            pAp = torch.dot(p, Ap)
            
            # avoid division by zero
            if pAp <= 0:
                break
                
            alpha = rdotr / pAp
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            
            # check convergence
            if new_rdotr < tol:
                break
                
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def update_policy(self):
        # check for empty episode
        if len(self.rewards) == 0:
            return 0.0
            
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (
            returns.std() + 1e-9
        )  # normalize returns

        # compute policy gradient
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)

        probs = self.policy_net(states)  # get policy for all states in episode
        # add numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # extract log of policy for taken action a_t given s_t
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze())

        policy_gradient = torch.autograd.grad(
            (log_probs * returns).sum(),
            # standard gradient. computes grad_theta summation of t log pie(a_t|s_t) * G_t. this is the standard reinforce gradient
            self.policy_net.parameters(),
        )

        # flatten to single vector
        policy_gradient = torch.cat([grad.view(-1) for grad in policy_gradient])

        # compute natural gradient using conjugate gradient
        # solves F * x = grad J_theta for x. x is the natural gradient direciton. uses conjugate grad to avoid computing F^-1
        natural_gradient = self.conjugate_gradient(policy_gradient)

        # better parameter update with NaN checking
        old_params = self.get_flat_params()
        new_params = old_params + natural_gradient * self.step_size
        
        # Check for NaN values
        if torch.isnan(new_params).any():
            print("Warning: NaN detected in natural gradient update, skipping...")
            grad_norm = 0.0
        else:
            # update parameters
            self.set_flat_params(new_params)
            grad_norm = torch.norm(natural_gradient).item()

        # clear episode data
        self.states = []
        self.actions = []
        self.rewards = []

        return grad_norm


def train_policy_gradient(
    agent_class, env_name="CartPole-v1", num_episodes=1000, **kwargs
):
    """generic training function for policy gradient agents"""
    env = gym.make(env_name)

    if hasattr(env.action_space, "n"):  # check if discrete actions
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = agent_class(state_size, action_size, **kwargs)
    else:  # continuous
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        agent = agent_class(state_size, action_size, **kwargs)

    scores = []
    running_scores = []  # track running average

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.store_reward(reward)
            total_reward += reward

            if terminated or truncated:
                break

            state = next_state

        # update policy
        loss = agent.update_policy()
        scores.append(total_reward)
        running_scores.append(total_reward)
        
        # keep only last 100 scores for running average
        if len(running_scores) > 100:
            running_scores.pop(0)

        if episode % 100 == 0:
            avg_score = np.mean(running_scores)  # use running average
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
            
            # check if CartPole is solved (195+ average)
            if avg_score >= 195.0:
                print(f"Environment solved in {episode} episodes!")
                break

    env.close()
    return agent, scores


if __name__ == "__main__":
    print("Training REINFORCE with Baseline...")
    agent, scores = train_policy_gradient(REINFORCEWithBaseline, "CartPole-v1", lr=0.001)

    print("Training Natural Policy Gradient...")
    try:
        agent_npg, scores_npg = train_policy_gradient(NaturalPolicyGradient, "CartPole-v1", lr=0.01, damping=0.01)
    except Exception as e:
        print(f"NPG training failed: {e}")
