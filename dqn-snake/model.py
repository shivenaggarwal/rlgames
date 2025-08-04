import os
from numpy.ma import masked_not_equal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNet(nn.Module):  # simple nn for approxing q values in deep q-learning
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.relu(self.l1(x)))

    
    def save(self, file_name: str="model.pth") -> None:
        model_f_path = './model'
        if not os.path.exists(model_f_path):
            os.makedirs(model_f_path)

        file_path = os.path.join(model_f_path, file_name)
        torch.save(self.state_dict(), file_path)


class QNetTrainer: # to perform gradient based updates for q network using dqn algo
    def __init__(self, model:QNet, lr:float, gamma: float):
        self.model = model 
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # reshape inputs if only one sample is provided (non-batch input)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )


        # compute predicted q values from current state
        pred = self.model(state)

        # clone predictions to use as targets
        target = pred.clone()

        # compute updated q values using bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # backprop and optim
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


