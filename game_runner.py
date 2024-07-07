from game import SnakeGame, GameState, DirectionChange, Direction
from game_feature_extractor import GameFeatureExtractor

if __name__ == '__main__':
    game = SnakeGame(speed=5)
    game_feature_extractor = GameFeatureExtractor(game=game)

    while True:
        if game.game_state == GameState.RUNNING:
            reward, done = game.tick()

            direction = game.direction

            is_danger_leftwards = game.is_collision(
                position=game.snake_pos.plus_direction(DirectionChange.LEFTWARDS.applied_to(direction))
            )

            is_danger_ahead = game.is_collision(
                position=game.snake_pos.plus_direction(DirectionChange.NONE.applied_to(direction))
            )

            is_danger_rightwards = game.is_collision(
                position=game.snake_pos.plus_direction(DirectionChange.RIGHTWARDS.applied_to(direction))
            )

            is_moving_left = game.direction == Direction.LEFT
            is_moving_up = game.direction == Direction.UP
            is_moving_right = game.direction == Direction.RIGHT
            is_moving_down = game.direction == Direction.DOWN

            is_food_left = game.food_pos.x < game.snake_pos.x
            is_food_up = game.food_pos.y < game.snake_pos.y
            is_food_right = game.food_pos.x > game.snake_pos.x
            is_food_down = game.food_pos.y > game.snake_pos.y

            food_text = ""

            if is_food_left:
                food_text += "Left;"
            elif is_food_right:
                food_text += "Right;"

            if is_food_up:
                food_text += "Up;"
            elif is_food_down:
                food_text += "Down;"

            if is_danger_ahead:
                print("DANGER AHEAD")

            if is_danger_leftwards:
                print("DANGER LEFTWARDS")

            if is_danger_rightwards:
                print("DANGER RIGHTWARDS")

            if done:
                game.game_state = GameState.STOPPED
                game.display_game_over()