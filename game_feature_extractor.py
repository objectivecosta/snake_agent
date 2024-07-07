import numpy as np

from game import SnakeGame, DirectionChange, Direction

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

    def linear_inputs(self) -> np.array:
        game = self.game

        direction: Direction = game.direction

        is_danger_leftwards = game.is_collision(
            position=self.game.snake_pos.plus_direction(DirectionChange.LEFTWARDS.applied_to(direction))
        )

        is_danger_ahead = game.is_collision(
            position=self.game.snake_pos.plus_direction(DirectionChange.NONE.applied_to(direction))
        )

        is_danger_rightwards = game.is_collision(
            position=self.game.snake_pos.plus_direction(DirectionChange.RIGHTWARDS.applied_to(direction))
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
        ], dtype=int)
