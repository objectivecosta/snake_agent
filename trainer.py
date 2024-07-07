from collections import deque
import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from game import Direction, SnakeGame, GameState, Position, DirectionChange
from game_feature_extractor import GameFeatureExtractor
from model_linear import SnakeLinearNetwork
import matplotlib.pyplot as plt
from IPython import display as display_ipyt

MEMORY_SIZE = 100_000
BATCH_SIZE = 2048
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.985
EPSILON_REDUCTION = 0.0025
GAMMA = 0.9
TARGET_UPDATE_FREQ = 25
LR = 0.001

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
        # plt.pause(.1)


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
        batch = self.memory

        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)

        return [(element.state, element.action, element.reward, element.next_state, element.done) for element in batch]


# noinspection DuplicatedCode
class Trainer:
    def __init__(self, game: SnakeGame, gamma=GAMMA):
        self.game = game
        self.game_feature_extractor = GameFeatureExtractor(game)

        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)

        self.epsilon = 1.0
        self.gamma = gamma
        self.primary_network = SnakeLinearNetwork()
        self.target_network = SnakeLinearNetwork()

        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=LR)

        self.logger = TrainingLogger()

    def select_action(self, state, epsilon) -> DirectionChange:
        all_changes = [
            DirectionChange.LEFTWARDS,  # 0
            DirectionChange.NONE,  # 1
            DirectionChange.RIGHTWARDS,  # 2
        ]

        state_tensor = torch.tensor(state, dtype=torch.float32)

        if random.random() < epsilon:
            return all_changes[random.randint(0, 2)]  # Assuming 3 actions
        else:
            with torch.no_grad():
                return all_changes[self.primary_network(state_tensor).argmax().item()]

    def q_learning(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        q_value = self.primary_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # next_q_value = self.target_network(next_state).max(1)[0]
        next_q_value = self.primary_network(next_state).max(1)[0]
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Compute loss with importance-sampling weights
        loss = F.mse_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def q_learning_alt(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 1: predicted Q values with current state
        pred = self.primary_network(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.primary_network(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Compute loss with importance-sampling weights
        loss = F.mse_loss(target, pred)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def q_learning_alt2(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.primary_network(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.primary_network(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()

        loss = F.mse_loss(target, pred)
        loss.backward()

        self.optimizer.step()

    def long_term_learning(self):
        batch = self.replay_buffer.sample(BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        self.q_learning_alt2(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def train(self):
        idx = 0
        while True:
            # state = self.game_feature_extractor.game_layers()
            state = self.game_feature_extractor.linear_inputs()
            action = self.select_action(state=state, epsilon=self.epsilon)

            self.game.change_to = action.applied_to(self.game.direction)
            reward, done = self.game.tick()
            next_state = self.game_feature_extractor.linear_inputs()

            self.replay_buffer.push(Replay(state, action.array_from_action(), reward, next_state, done))
            self.q_learning_alt2(state, action.array_from_action(), reward, next_state, done)

            if done:
                self.long_term_learning()
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
            print("Epsilon: {}".format(self.epsilon))
            print("Updating target network")
            self.target_network.load_state_dict(self.primary_network.state_dict())


if __name__ == '__main__':
    game = SnakeGame(speed=500, game_width=48, game_height=48, is_agent_playing=True)
    trainer = Trainer(game=game, gamma=GAMMA)
    trainer.train()
    # game.game_loop()
