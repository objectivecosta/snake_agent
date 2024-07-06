from collections import deque
import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from game import Direction, SnakeGame, GameState, Position
from model_linear import SnakeNetwork
import matplotlib.pyplot as plt
from IPython import display as display_ipyt

BATCH_SIZE = 256
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10


class TrainingLogger:
    def __init__(self):
        self.total_score = 0
        self.record = 0
        self.tick_counts = []
        self.scores = []
        self.averages = []

    def append(self, tick_count, result):
        self.tick_counts.append(tick_count)
        self.scores.append(result)
        self.averages.append(np.sum(self.scores) / len(self.scores))

    def display(self):
        average_tick_span = np.sum(self.tick_counts) / len(self.tick_counts)
        print("Average tick-span of game: {}".format(average_tick_span))

        display_ipyt.clear_output(wait=True)
        display_ipyt.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(self.scores)
        plt.plot(self.averages)
        plt.ylim(ymin=0)
        plt.text(len(self.scores) - 1, self.scores[-1], str(self.scores[-1]))
        plt.text(len(self.averages) - 1, self.averages[-1], str(self.averages[-1]))
        plt.show(block=False)
        plt.pause(.1)


class Replay:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, replay: Replay):
        self.memory.append(replay)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = [(element.state, element.action, element.reward, element.next_state, element.done) for element in batch]
        return batch


class GameFeatureExtractor:
    def __init__(self, game: SnakeGame):
        self.game = game

    def game_layers(self):
        snake_view = self.snake_layer(self.game)
        food_view = self.food_layer(self.game)

        return np.stack([
            snake_view,
            food_view
        ], axis=0)

    def snake_layer(self, game: SnakeGame) -> np.array:
        result = np.zeros((game.width, game.height))

        for pos in game.snake_body:
            if pos.x < game.width and pos.y < game.height:
                result[pos.x][pos.y] = 1

        return result

    def food_layer(self, game: SnakeGame) -> np.array:
        result = np.zeros((game.width, game.height))
        result[game.food_pos.x][game.food_pos.y] = 1
        return result

    def linear_inputs(self, game: SnakeGame) -> np.array:

        direction: Direction = game.direction

        is_danger_leftwards = game.is_collision(
            position=self.position_plus_direction(self.game.snake_pos, self.shift_left(direction))
        )

        is_danger_ahead = game.is_collision(
            position=self.position_plus_direction(self.game.snake_pos, direction)
        )

        is_danger_rightwards = game.is_collision(
            position=self.position_plus_direction(self.game.snake_pos, self.shift_right(direction))
        )

        is_moving_left = direction == Direction.LEFT
        is_moving_up = direction == Direction.UP
        is_moving_right = direction == Direction.RIGHT
        is_moving_down = direction == Direction.DOWN

        is_food_left = game.food_pos.x < game.snake_pos.x
        is_food_up = game.food_pos.y < game.snake_pos.y
        is_food_right = game.food_pos.x > game.snake_pos.x
        is_food_down = game.food_pos.y > game.snake_pos.y

        return np.array([
            is_danger_leftwards,
            is_danger_ahead,
            is_danger_rightwards,

            is_moving_left,
            is_moving_up,
            is_moving_right,
            is_moving_down,

            is_food_left,
            is_food_up,
            is_food_right,
            is_food_down
        ])

    def position_plus_direction(self, position: Position, direction: Direction) -> Position:
        if direction == Direction.LEFT:
            return Position(position.x - 1, position.y)
        elif direction == Direction.UP:
            return Position(position.x, position.y - 1)
        elif direction == Direction.RIGHT:
            return Position(position.x + 1, position.y)
        elif direction == Direction.DOWN:
            return Position(position.x, position.y + 1)

    def shift_left(self, direction: Direction) -> Direction:
        if direction == Direction.LEFT:
            return Direction.DOWN
        elif direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.RIGHT:
            return Direction.UP
        elif direction == Direction.DOWN:
            return Direction.RIGHT

    def shift_right(self, direction: Direction) -> Direction:
        if direction == Direction.LEFT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.RIGHT
        elif direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.LEFT


class Trainer:
    def __init__(self, game: SnakeGame, gamma=0.99):
        self.game = game
        self.game_feature_extractor = GameFeatureExtractor(game)

        self.replay_buffer = ReplayBuffer(10000)

        self.epsilon = 1.0
        self.gamma = gamma
        self.primary_network = SnakeNetwork(game.width, game.height)
        self.target_network = SnakeNetwork(game.width, game.height)

        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=1e-3)

        self.logger = TrainingLogger()

    def select_action(self, state, epsilon) -> Direction:
        all_directions = [
            Direction.LEFT,  # 0
            Direction.UP,  # 1
            Direction.RIGHT,  # 2
            Direction.DOWN,  # 3
        ]

        state_tensor = torch.tensor(state, dtype=torch.float32)

        if random.random() < epsilon:
            return all_directions[random.randint(0, 3)]  # Assuming 4 actions
        else:
            with torch.no_grad():
                return all_directions[self.primary_network(state_tensor).argmax().item()]

    def q_learning(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        q_value = self.primary_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = self.target_network(next_state).max(1)[0]
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Compute loss with importance-sampling weights
        loss = F.mse_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample_and_learn(self):
        if len(self.replay_buffer.memory) < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        self.q_learning(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def train(self):
        idx = 0
        while True:
            state = self.game_feature_extractor.game_layers()
            action = self.select_action(state, self.gamma)

            self.game.change_to = action
            reward, done = self.game.tick()
            next_state = self.game_feature_extractor.game_layers()

            self.replay_buffer.push(Replay(state, action, reward, next_state, done))

            if done:
                idx += 1
                self.logger.append(self.game.tick_count, self.game.score)
                self.game.restart()

                if idx % 50 == 0:
                    self.logger.display()
                self.perform_updates(idx)

    def perform_updates(self, idx=0):
        # Update epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        # Update target network
        if idx % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.primary_network.state_dict())


if __name__ == '__main__':
    game = SnakeGame(speed=150, game_width=24, game_height=24, is_agent_playing=True)
    trainer = Trainer(game=game, gamma=0.99)
    trainer.train()
    # game.game_loop()
