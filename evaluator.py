import os
import time

import torch

from game import SnakeGame, DirectionChange
from game_feature_extractor import GameFeatureExtractor
from model_linear import SnakeLinearNetwork


class Evaluator:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.game_feature_extractor = GameFeatureExtractor(game)

        self.network_under_evaluation = SnakeLinearNetwork()
        self.network_under_evaluation.read_from_disk()

    def select_action(self, state) -> DirectionChange:
        all_changes = [
            DirectionChange.LEFTWARDS,  # 0
            DirectionChange.NONE,  # 1
            DirectionChange.RIGHTWARDS,  # 2
        ]

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            return all_changes[self.network_under_evaluation(state_tensor).argmax().item()]

    def evaluate(self):
        while True:
            state = self.game_feature_extractor.linear_inputs()
            action = self.select_action(state=state)

            self.game.change_to = action.applied_to(self.game.direction)
            _,  done = self.game.tick()

            if done:
                time.sleep(10)
                self.game.restart()

if __name__ == '__main__':
    # We say agent is not playing, even though agent is actually playing. This is intentional.
    game = SnakeGame(speed=75, game_width=48, game_height=48, is_agent_playing=False)
    evaluator = Evaluator(game)
    evaluator.evaluate()

