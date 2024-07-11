from collections import deque
import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from game import Direction, SnakeGame, GameState, Position, DirectionChange
from game_feature_extractor import GameFeatureExtractor
from model_cnn import SnakeCNNNetwork
import matplotlib.pyplot as plt
from IPython import display as display_ipyt

MEMORY_SIZE = 100_000
BATCH_SIZE = 2048
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.985
EPSILON_REDUCTION = 0.0025
GAMMA = 0.9
TARGET_UPDATE_FREQ = 25
TARGET_SAVE_FREQ = 100
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
    def __init__(self, state, linear_state, action, reward, next_state, next_linear_state, done):
        self.state = state
        self.linear_state = linear_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_linear_state = next_linear_state
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

        return [(element.state, element.linear_state, element.action, element.reward, element.next_state, element.next_linear_state, element.done) for element in batch]


# noinspection DuplicatedCode
class Trainer:
    def __init__(self, game: SnakeGame, gamma=GAMMA):
        self.game = game
        self.game_feature_extractor = GameFeatureExtractor(game)

        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)

        self.epsilon = 1.0
        self.gamma = gamma
        self.primary_network = SnakeCNNNetwork(grid_width=game.width, grid_height=game.height)
        self.target_network = SnakeCNNNetwork(grid_width=game.width, grid_height=game.height)

        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=LR)

        self.logger = TrainingLogger()

    def select_action(self, state, linear_state, epsilon) -> DirectionChange:
        all_changes = [
            DirectionChange.LEFTWARDS,  # 0
            DirectionChange.NONE,  # 1
            DirectionChange.RIGHTWARDS,  # 2
        ]

        state_tensor = torch.tensor(state, dtype=torch.float32)
        linear_state_tensor = torch.tensor(linear_state, dtype=torch.float32)

        if random.random() < epsilon:
            return all_changes[random.randint(0, 2)]  # Assuming 3 actions
        else:
            with torch.no_grad():
                index_to_select = self.primary_network(state_tensor, linear_state_tensor).argmax().item()
                return all_changes[index_to_select]

    def q_learning(self, state, linear_state, action, reward, next_state, next_linear_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        linear_state = torch.tensor(linear_state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_linear_state = torch.tensor(next_linear_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        if state.dim() == 3:
            state = state.unsqueeze(0)
            linear_state = linear_state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            next_linear_state = next_linear_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Reshaping one-dimensional action to be a 1, 1 vector
        action_unsqueezed = action.unsqueeze(1)

        # Fetching network estimation for this state (estimation of which action)
        primary_network_value = self.primary_network(state, linear_state)

        # Gathering, from the network, what Q value it estimated for the action we took
        # (i.e. if action was 0, we gather the 0-th element. This needs to be 1 dimensioned because we want to
        # get the inner value.)
        pre_q_value = primary_network_value.gather(1, action_unsqueezed)

        # We squeeze it back into a 1D value, so we can actually do simple math with it.
        q_value = pre_q_value.squeeze(1)

        # Now we make an estimation of the next q_value, given the state we end up with.
        next_q_value = self.target_network(next_state, next_linear_state).max(1)[0]

        # And now we do the Bellman equation to actually see the Q value of this action,
        # based on the reward + the next state.
        target_q_value = reward + (1 - done) * self.gamma * next_q_value

        # The model predicted some Q value. We estimate, via Bellman equations, another one.
        # We need to start working on minimizing this loss to train the model.

        # Compute loss with importance-sampling weights
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_value, target_q_value)
        loss.backward()
        self.optimizer.step()

    def long_term_learning(self):
        batch = self.replay_buffer.sample(BATCH_SIZE)
        state_batch, linear_state_batch, action_batch, reward_batch, next_state_batch, next_linear_state_batch, done_batch = zip(*batch)

        self.q_learning(state_batch, linear_state_batch, action_batch, reward_batch, next_state_batch, next_linear_state_batch, done_batch)

    def train(self):
        idx = 0
        while True:
            state = self.game_feature_extractor.game_layers()
            linear_state = self.game_feature_extractor.linear_inputs()
            action = self.select_action(state=state, linear_state=linear_state, epsilon=self.epsilon)

            self.game.change_to = action.applied_to(self.game.direction)
            reward, done = self.game.tick()

            next_state = self.game_feature_extractor.game_layers()
            next_linear_state = self.game_feature_extractor.linear_inputs()

            self.replay_buffer.push(Replay(state, linear_state, action.index_from_action(), reward, next_state, next_linear_state, done))
            self.q_learning(
                state=state,
                linear_state=linear_state,
                action=action.index_from_action(),
                reward=reward,
                next_state=next_state,
                next_linear_state=next_linear_state,
                done=done
            )

            if done:
                self.long_term_learning()
                idx += 1
                self.logger.append(self.game.tick_count, self.game.score)

                if idx > 1000 and self.logger.averages[-1] > self.game.score:
                    print("Slowing down, training is good enough...")
                    self.game.speed = 75

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
        if idx % TARGET_SAVE_FREQ == 0:
            self.target_network.write_to_disk()
            pass


if __name__ == '__main__':
    game = SnakeGame(speed=500, game_width=32, game_height=32, is_agent_playing=True)
    trainer = Trainer(game=game, gamma=GAMMA)
    trainer.train()
    # game.game_loop()
