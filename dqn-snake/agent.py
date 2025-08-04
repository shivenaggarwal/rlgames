import torch
import numpy as np
import random
from env import SnakeGame, Direction, Point
from collections import deque
from model import QNet, QNetTrainer
import time
import matplotlib.pyplot as plt
from IPython import display

# hyperparameters
MAX_MEMORY = 100_000  # to store 100_000 items in the memory
BATCH_SIZE = 1000
LR = 0.001

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # exploration rate
        self.gamma = 0.9  # discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # experience replay buffer
        self.model = QNet(input_size=11, hidden_size=256, output_size=3)  # dqn model
        self.trainer = QNetTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        pt_l = Point(head.x - 20, head.y)
        pt_r = Point(head.x + 20, head.y)
        pt_u = Point(head.x, head.y - 20)
        pt_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(pt_r))
            or (dir_l and game.is_collision(pt_l))
            or (dir_u and game.is_collision(pt_u))
            or (dir_d and game.is_collision(pt_d)),
            # danger right
            (dir_u and game.is_collision(pt_r))
            or (dir_d and game.is_collision(pt_l))
            or (dir_l and game.is_collision(pt_u))
            or (dir_r and game.is_collision(pt_d)),
            # danger left
            (dir_d and game.is_collision(pt_r))
            or (dir_u and game.is_collision(pt_l))
            or (dir_r and game.is_collision(pt_u))
            or (dir_l and game.is_collision(pt_d)),
            # current movement direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location relative to head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    # stores agent experience in memory
    def remember(self, state, action, reward, next_state, done):
        # pop left if max memory is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, done)

    def train_short_memory(
        self, state, action, reward, next_state, done
    ):  # online learning
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games  # exploration decrease exploration over time
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        else:
            # exploitation choose the best action
            state0 = torch.tensor(state, dtype=torch.float)
            move = torch.argmax(self.model(state0)).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        state_old = agent.get_state(game)  # get current state

        final_move = agent.get_action(state_old) # get next move

        # perform next move and get reward/state info
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory and store experience
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # reset game and train long memory. the experience replay
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # save best model
            if score > best_score:
                best_score = score
                agent.model.save()

            print("Game: ", agent.n_games, "Score: ", score, "Record:", best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = sum(plot_scores) / len(plot_scores)
            plot_avg_scores.append(mean_score)
            plot(plot_scores, plot_avg_scores)

            time.sleep(0.1)


train()
