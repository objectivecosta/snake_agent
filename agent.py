from game import SnakeGame, Direction
import numpy as np

class Agent:
    def __init__(self):
        pass

    def output(self, game: SnakeGame) -> Direction:
        snake_view = self.snake_layer(game)
        food_view = self.food_layer(game)

        layers = np.stack([
            snake_view,
            food_view
        ], axis=0)

        # TODO: Pass both layers to a nn.Conv2d layer

        # TODO: Potentially pass a cheat-code (i.e valid actions) to a Linear layer later in the model.

        return Direction.RIGHT # Dumb default

    def snake_layer(self, game: SnakeGame) -> np.array:
        result = np.zeros((game.width, game.height))

        for pos in game.snake_body:
            result[pos.x][pos.y] = 1

        return result

    def food_layer(self, game: SnakeGame) -> np.array:
        result = np.zeros((game.width, game.height))
        result[game.food_pos.x][game.food_pos.y] = 1
        return result


