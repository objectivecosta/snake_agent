"""
Snake Eater
Made with PyGame
"""
import enum
import math
from typing import Self

import sys, random
import pygame

BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)
RED_COLOR = pygame.Color(255, 0, 0)
GREEN_COLOR = pygame.Color(0, 255 , 0)
BLUE_COLOR = pygame.Color(0, 0, 255)

type MoveResult = tuple[float, bool]


class GameState(enum.Enum):
    STOPPED = 0
    RUNNING = 1


class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def is_valid_change(self, to_direction: Self) -> bool:
        if self == to_direction:
            return False

        # Can't switch to opposite directions:
        if self == Direction.UP and to_direction == Direction.DOWN:
            return False
        elif self == Direction.DOWN and to_direction == Direction.UP:
            return False
        elif self == Direction.LEFT and to_direction == Direction.RIGHT:
            return False
        elif self == Direction.RIGHT and to_direction == Direction.LEFT:
            return False

        return True


class DirectionChange(enum.Enum):
    LEFTWARDS = 0
    NONE = 1
    RIGHTWARDS = 2

    def applied_to(self, direction: Direction) -> Direction:
        if self == DirectionChange.LEFTWARDS:
            return self._shift_left(direction)
        elif self == DirectionChange.RIGHTWARDS:
            return self._shift_right(direction)
        else:
            return direction

    def index_from_action(self) -> int:
        return self.value

    def array_from_action(self) -> list:
        list = [0,0,0]
        list[self.index_from_action()] = 1
        return list

    @staticmethod
    def _shift_left(direction: Direction) -> Direction:
        if direction == Direction.LEFT:
            return Direction.DOWN
        elif direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.RIGHT:
            return Direction.UP
        elif direction == Direction.DOWN:
            return Direction.RIGHT

    @staticmethod
    def _shift_right(direction: Direction) -> Direction:
        if direction == Direction.LEFT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.RIGHT
        elif direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.LEFT


class Position:
    def __init__(self, x: int, y: int, point_size: int = 10):
        self.x = x
        self.y = y
        self.point_size = point_size

    def as_tuple(self) -> tuple:
        return self.x, self.y

    def as_pixel_tuple(self):
        return self.x * self.point_size, self.y * self.point_size

    def plus_direction(self, direction: Direction) -> Self:
        if direction == Direction.LEFT:
            return Position(self.x - 1, self.y)
        elif direction == Direction.UP:
            return Position(self.x, self.y - 1)
        elif direction == Direction.RIGHT:
            return Position(self.x + 1, self.y)
        elif direction == Direction.DOWN:
            return Position(self.x, self.y + 1)

    def __eq__(self, other: Self) -> bool:
        return self.as_tuple() == other.as_tuple()

    def clone(self):
        return Position(self.x, self.y, self.point_size)


class SnakeGame:
    def __init__(self, speed=25, game_width=48, game_height=48, is_agent_playing=False):
        self.point_size = 10
        self.is_agent_playing = is_agent_playing

        # Difficulty settings
        # Easy      ->  10
        # Medium    ->  25
        # Hard      ->  40
        # Harder    ->  60
        # Impossible->  120
        self.game_state = GameState.RUNNING
        self.change_to: Direction = Direction.RIGHT
        self.direction: Direction = Direction.RIGHT
        self.food_is_visible: bool = False
        self.food_pos = None
        self.snake_body = None
        self.snake_pos = None
        self.fps_controller = None
        self.game_window = None
        self.speed = speed

        # Point size
        self.width = game_width
        self.height = game_height

        # Window size
        self.frame_size_x = self.width * self.point_size
        self.frame_size_y = self.height * self.point_size

        # Init PyGame
        check_errors = pygame.init()
        if check_errors[1] > 0:
            print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
            sys.exit(-1)
        else:
            print('[+] Game successfully initialised')

        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        self.game_window.fill(BLACK_COLOR)

        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()

        # Init Snake
        self.snake_pos = Position(0, 0, self.point_size)
        self.snake_body = [
            Position(0, 0, self.point_size),
        ]

        # Init Food
        self.food_pos = Position(0, 0, self.point_size)
        self.food_is_visible = False

        self.score = 0
        self.tick_count = 0

        self.restart()

    def restart(self):
        self.snake_pos = Position(10, 5, self.point_size)
        self.snake_body = [
            Position(10, 5, self.point_size),
            Position(9, 5, self.point_size),
            Position(8, 5, self.point_size),
        ]

        # Init Food
        self.food_pos = self.random_position()
        self.food_is_visible = True

        self.score = 0
        self.tick_count = 0

    def random_position(self) -> Position:
        return Position(
            random.randrange(1, self.width),
            random.randrange(1, self.height),
            self.point_size
        )

    def receive_keyboard_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Whenever a key is pressed down
            elif event.type == pygame.KEYDOWN and self.is_agent_playing == False:
                # W -> Up; S -> Down; A -> Left; D -> Right
                if event.key == pygame.K_UP or event.key == ord('w'):
                    self.change_to = Direction.UP
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.change_to = Direction.DOWN
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.change_to = Direction.LEFT
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.change_to = Direction.RIGHT
                # Esc -> Create event to quit the game
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

    def tick(self) -> MoveResult:
        self.tick_count += 1
        reward = 0.0
        done = False

        self.receive_keyboard_events()

        is_danger_ahead = self.is_collision(
            position=self.snake_pos.plus_direction(DirectionChange.NONE.applied_to(self.direction))
        )

        if self.change_to == self.direction and not is_danger_ahead:
            reward += 0.1 # Incentivize stability
        elif self.change_to != self.direction and not is_danger_ahead:
            reward -= 0.1 # De-incentivize useless changes

        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.direction.is_valid_change(self.change_to):
            self.direction = self.change_to

        distance_to_food_before = math.sqrt(
            pow(self.food_pos.x - self.snake_pos.x, 2) + pow(self.food_pos.y - self.snake_pos.y, 2)
        )

        self.snake_pos = self.snake_pos.clone()

        # Moving the snake
        if self.direction == Direction.UP:
            self.snake_pos.y -= 1
        if self.direction == Direction.DOWN:
            self.snake_pos.y += 1
        if self.direction == Direction.LEFT:
            self.snake_pos.x -= 1
        if self.direction == Direction.RIGHT:
            self.snake_pos.x += 1

        distance_to_food_after = math.sqrt(
            pow(self.food_pos.x - self.snake_pos.x, 2) + pow(self.food_pos.y - self.snake_pos.y, 2)
        )

        # Snake body growing mechanism
        self.snake_body.insert(0, self.snake_pos)
        if self.snake_pos == self.food_pos:
            reward += 1.0
            self.score += 1
            self.food_is_visible = False
        else:
            if distance_to_food_before > distance_to_food_after:
                # We got closer. Incentivize this behavior:
                reward += 0.25
            else:
                reward -= 0.25

            self.snake_body.pop()

        # Spawning food on the screen
        if not self.food_is_visible:
            self.food_pos = self.random_position()
        self.food_is_visible = True

        # GFX
        self.game_window.fill(BLACK_COLOR)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, GREEN_COLOR,
                             pygame.Rect(pos.as_pixel_tuple()[0], pos.as_pixel_tuple()[1], self.point_size,
                                         self.point_size))

        # Snake food
        pygame.draw.rect(self.game_window, WHITE_COLOR,
                         pygame.Rect(self.food_pos.as_pixel_tuple()[0], self.food_pos.as_pixel_tuple()[1],
                                     self.point_size, self.point_size))

        # Game Over conditions

        # Touching the snake body / edges
        if self.is_collision(self.snake_pos):
            reward = -1.0
            done = True

        self.display_score(False, WHITE_COLOR, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate

        if not self.is_agent_playing:
            self.fps_controller.tick(self.speed)
        else:
            if self.speed < 200:
                self.fps_controller.tick(self.speed)

        return reward, done

    def is_collision(self, position: Position) -> bool:
        for block in self.snake_body[1:]:
            if position.x == block.x and position.y == block.y:
                return True

        if position.x < 0 or position.x >= self.width:
            return True
        if position.y < 0 or position.y >= self.height:
            return True

        return False

    def game_loop(self):
        if self.is_agent_playing:
            return

        while True:
            if self.game_state == GameState.RUNNING:
                _, done = self.tick()
                if done:
                    self.game_state = GameState.STOPPED
                    self.display_game_over()
            else:
                pass

    def display_score(self, is_end_game, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if is_end_game:
            score_rect.midtop = (self.frame_size_x / 10, 15)
        else:
            score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
        self.game_window.blit(score_surface, score_rect)
        pygame.display.flip()

    def display_game_over(self):
        my_font = pygame.font.SysFont('times new roman', 90)
        game_over_surface = my_font.render('YOU DIED', True, RED_COLOR)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 4)
        self.game_window.fill(BLACK_COLOR)
        self.game_window.blit(game_over_surface, game_over_rect)
        self.display_score(True, RED_COLOR, 'times', 20)
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGame()
    game.game_loop()
