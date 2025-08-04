import random
import pygame
from enum import Enum
from collections import namedtuple
import numpy as np

# init pygame module and font
pygame.init()
font = pygame.font.SysFont("arial", 25)


# movement directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# snake coordinates
Point = namedtuple("Point", "x, y")

# color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 100, 0)
BLACK = (0, 0, 0)

# game constants
BLOCK_SIZE = 20
SPEED = 500


class SnakeGame:
    def __init__(self, w: int = 640, h: int = 480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("snake")
        self.clock = pygame.time.Clock()
        self.reset()
        self.last_positions = []
        self.starve_counter = 0

    def reset(self):  # reset the game to its original state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_positions = []
        self.starve_counter = 0

    def _place_food(self):  # randomly place food on board avoiding snake's body
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):  # executed one step given an action
        self.frame_iteration += 1

        # handle game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        prev_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.0
        game_over = False

        # reward/ penalty points
        step_penalty = -0.05
        closer_reward = 0.2
        farther_penalty = -0.2
        eat_reward = 10
        die_penalty = -10
        loop_penalty = 0
        starvation_penalty = 0
        starvation_limit = 100 * len(self.snake)
        starvation_penalty_value = -5
        loop_window = 12

        # detect loops (repeating head positions)
        self.last_positions.append(self.head)
        if len(self.last_positions) > loop_window:
            self.last_positions.pop(0)
        if self.last_positions.count(self.head) > 1:
            loop_penalty = -0.5

        # starvation detection
        self.starve_counter += 1
        if self.starve_counter > starvation_limit:
            starvation_penalty = starvation_penalty_value
            game_over = True
            reward = die_penalty + starvation_penalty + loop_penalty + step_penalty
            return reward, game_over, self.score

        # collision check or time out
        if self.is_collision() or self.frame_iteration > 200 * len(self.snake):
            game_over = True
            reward = die_penalty + loop_penalty + step_penalty
            return reward, game_over, self.score

        # reward based on food distance
        new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        if new_dist < prev_dist:
            reward += closer_reward

        elif new_dist > prev_dist:
            reward += farther_penalty

        else:
            reward += step_penalty

        # eating food
        if self.head == self.food:
            self.score += 1
            reward = eat_reward
            self._place_food()
            self.starve_counter = 0

        else:
            self.snake.pop()

        reward += loop_penalty + starvation_penalty
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    # checks if a given pt collides with snake or wall
    def is_collision(self, pt=None) -> bool:
        if pt is None:
            pt = self.head

        # wall collision
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True

        # self collision
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):  # renders the game frame
        self.display.fill(BLACK)

        # draw snake
        for pt in self.snake:
            pygame.draw.rect(
                self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        # draw food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        # draw score
        text = font.render("score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(
        self, action
    ):  # updates snakes direction and pos based on the given direction
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        # determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]

        else:
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir

        # new head position
        x, y = self.head.x, self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
