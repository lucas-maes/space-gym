from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.error import DependencyNotInstalled
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams, Steering
from gym_space.helpers import angle_to_unit_vector, vector_to_angle
import numpy as np

from gym_space.dynamic_model import ShipState, ship_vector_field

import math

MAX_SCREEN_SIZE = 600
SHIP_BODY_RADIUS = 15


def rotate_point(px, py, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (px * c - py * s, px * s + py * c)

def transform_point(p, x, y, ang):
    rx, ry = rotate_point(p[0], p[1], ang)
    return (x + rx, y + ry)

def transform_points(points, x, y, ang):
    return [transform_point(p, x, y, ang) for p in points]


@dataclass
class SpaceshipEnv(gym.Env, ABC):
    """Base class for all Spaceship environments and tasks

    Args:
        ship_params: parameters describing properties of the spaceship
        planets: list of parameters describing properties of the planets
        world_size: width and height of square 2D world
        max_abs_vel_angle: maximal absolute value of angular velocity
        step_size: number of seconds between consecutive observations
        vel_xy_std: approximate standard deviation of translational velocities
        with_lidar: include "lidars" for planets and goal (if present) in observations
        with_goal: if goal (point in space) is present in the environment
    """


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    ship_params: ShipParams
    planets: list[Planet]
    world_size: float
    max_abs_vel_angle: float
    step_size: float
    vel_xy_std: np.array
    with_lidar: bool
    with_goal: bool

    observation: np.array = field(init=False, default=None)
    last_action: np.array = field(init=False, default=None)
    last_xy: np.array = field(init=False, default=None)
    goal_pos: np.array = field(init=False, default=None)

    prev_ship_pos: deque = field(init=False, default=deque(maxlen=30))
    prev_pos_color_decay: float = field(init=False, default=0.85)
    render_mode: str = field(init=True, default="rgb_array")

    def __post_init__(self):
        self._init_observation_space()
        self._init_action_space()
        self._np_random = None
        self.seed()
        self._ship_state = ShipState(self.ship_params, self.planets, self.world_size, self.max_abs_vel_angle)
        self.world_translation = np.full(2, -self.world_size / 2)
        self.world_scale = MAX_SCREEN_SIZE  / self.world_size
        self.screen_size = self.world_scale * self.world_size    
        
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self._reset()
        assert self._ship_state.is_defined
        assert self.with_goal == (self.goal_pos is not None)
        self._make_observation()
        self.prev_ship_pos.clear()
        return self.observation, {}

    def step(self, raw_action):
        if isinstance(self.action_space, Box):
            raw_action = raw_action.astype(np.float32)
        assert self.action_space.contains(raw_action), raw_action
        action = np.array(self._translate_raw_action(raw_action))
        self.last_action = action
        self.last_xy = self._ship_state.pos_xy
        done = self._ship_state.step(action, self.step_size)
        self._make_observation()
        reward = self._reward()
        return self.observation, reward, done, False, {}

    def render(self, mode="human"):

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e


        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            else: # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_size, self.screen_size))


        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_size, self.screen_size))
        self.surf.fill((255, 251, 238))

        #### draw planets
        for planet in self.planets:
            x, y = self._world_to_screen(planet.center_pos)
            gfxdraw.filled_circle(self.surf, int(x), int(y), int(planet.radius * self.world_scale), (255, 249, 227))
            gfxdraw.aacircle(self.surf, int(x), int(y), int(planet.radius * self.world_scale), (138, 136, 128))

        #### draw ship
        # TODO DRAW and rotate everything with ship angle
        ship_screen_position = self._world_to_screen(self._ship_state.full_pos[:2])
        self.prev_ship_pos.append(ship_screen_position)

        x, y = ship_screen_position
        angle = self._ship_state.pos_angle

        #### draw ship trace
        opacity = 1.0
        for i in range(1, len(self.prev_ship_pos)):
            p0 = self.prev_ship_pos[-i]
            p1 = self.prev_ship_pos[-i - 1]
            a = max(0, min(255, int(opacity * 255)))
            color = (*(255, 100, 80), a)

            pygame.draw.line(self.surf, color, p0, p1, 1)

            opacity *= self.prev_pos_color_decay
            if a == 0:
                break

        # engine (local)
        engine_edge_length = SHIP_BODY_RADIUS * 1.7
        engine_width_angle = np.pi / 4
        engine_left_bottom_angle = -engine_width_angle / 2
        engine_right_bottom_angle = engine_width_angle / 2
        engine_left_bottom_pos = engine_edge_length * angle_to_unit_vector(engine_left_bottom_angle)
        engine_right_bottom_pos = engine_edge_length * angle_to_unit_vector(engine_right_bottom_angle)
        engine_poly_local = [
            (0.0, 0.0),
            (float(engine_left_bottom_pos[0]), float(engine_left_bottom_pos[1])),
            (float(engine_right_bottom_pos[0]), float(engine_right_bottom_pos[1])),
        ]
        # transform to screen
        engine_poly_world = transform_points(engine_poly_local, x, y, angle)
        engine_poly_world_int = [(int(px), int(py)) for px, py in engine_poly_world]
        gfxdraw.filled_polygon(self.surf, engine_poly_world_int, (138, 136, 128))



        # ship body
        gfxdraw.filled_circle(
            self.surf,
            int(x), int(y),
            int(SHIP_BODY_RADIUS),
            (255, 251, 238)
        )

        # ship body outline
        gfxdraw.aacircle(
            self.surf,
            int(x), int(y),
            int(SHIP_BODY_RADIUS),
            (138, 136, 128)
        )

        # ship middle
        gfxdraw.pixel(
            self.surf,
            int(x), int(y),
            (138, 136, 128)
        )


        # draw exhaust
        engine_width_angle = np.pi / 4
        exhaust_begin_radius = SHIP_BODY_RADIUS * 1.9
        exhaust_end_radius = SHIP_BODY_RADIUS * 2.2

        for flame_angle in np.linspace(-engine_width_angle / 4, engine_width_angle / 4, 3):
            vec = angle_to_unit_vector(flame_angle)

            p0_local = (float(exhaust_begin_radius * vec[0]), float(exhaust_begin_radius * vec[1]))
            p1_local = (float(exhaust_end_radius * vec[0]), float(exhaust_end_radius * vec[1]))

            p0 = transform_point(p0_local, x, y, angle)
            p1 = transform_point(p1_local, x, y, angle)

            gfxdraw.line(
                self.surf,
                int(p0[0]), int(p0[1]),
                int(p1[0]), int(p1[1]),
                (138, 136, 128)
            )

        #### draw goal
        if self.goal_pos is not None:
            x, y = (int(p) for p in self._world_to_screen(self.goal_pos))
            gfxdraw.line(
                self.surf,
                x + 10, y + 10, x - 10, y - 10,
                (34, 46, 80)
            )
            gfxdraw.line(
                self.surf,
                x + 10, y - 10, x - 10, y + 10,
                (34, 46, 80)
            )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _world_to_screen(self, world_pos: np.array):
        return self.world_scale * (world_pos - self.world_translation)

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def vector_field(self, raw_action, state_vec: np.array = None):
        if state_vec is None:
            state_vec = self._ship_state._state_vec
        action = np.array(self._translate_raw_action(raw_action))
        return ship_vector_field(self.ship_params, self.planets, action, 0.0, state_vec)

    def _init_observation_space(self):
        obs_high = [1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0]
        if self.with_lidar:
            # as normalized world is [-1, 1]^2, the highest distance between two points is 2 sqrt(2)
            # (x, y) vector for each planet
            obs_high += 2 * len(self.planets) * [2 * np.sqrt(2)]
            if self.with_goal:
                obs_high += 2 * [2 * np.sqrt(2)]
        obs_high = np.array(obs_high, dtype=np.float32)
        self.observation_space = Box(low=-obs_high, high=obs_high)

    def _make_observation(self):
        # make sure that x and y positions are between -1 and 1
        obs_pos_xy = self._ship_state.pos_xy  # / self.world_size
        # normalize translational velocity
        obs_vel_xy = self._ship_state.vel_xy  # / self.vel_xy_std
        # make sure that angular velocity is between -1 and 1
        obs_vel_angle = self._ship_state.vel_angle  # / self.max_abs_vel_angle
        # represent angle as cosine and sine
        angle = self._ship_state.pos_angle
        angle_repr = angle_to_unit_vector(angle)
        # angle_repr = np.array([np.cos(angle), np.sin(angle)])
        observation = [obs_pos_xy, angle_repr, obs_vel_xy, np.array([obs_vel_angle])]

        if self.with_lidar:
            observation += [self._create_lidar_vector(p.center_pos, p.radius) for p in self.planets]
            if self.with_goal:
                observation += [self._create_lidar_vector(self.goal_pos)]

        self.observation = np.concatenate(observation)

    def _create_lidar_vector(self, obj_pos: np.array, obj_radius: float = 0.0) -> np.array:
        """Create vector from ship to some object."""

        ship_center_obj_vec = obj_pos - self._ship_state.pos_xy
        ship_obj_angle = vector_to_angle(ship_center_obj_vec)  # - np.pi / 2 - self._ship_state.pos_angle
        ship_obj_angle %= 2 * np.pi
        scale = (np.linalg.norm(ship_center_obj_vec) - obj_radius) * 2 / self.world_size
        return angle_to_unit_vector(ship_obj_angle) * scale

    @property
    def planets_lidars(self):
        if not self.with_lidar:
            return None
        if self.with_lidar and not self.with_goal:
            return self.observation[-2 * len(self.planets) :].reshape(-1, 2)
        if self.with_lidar and self.with_goal:
            # the last two observations of state vec is the goal lidar
            return self.observation[-2 * (len(self.planets) + 1) : -2].reshape(-1, 2)

    @property
    def goal_lidar(self):
        if not (self.with_lidar and self.with_goal):
            return None
        return self.observation[-2:]

    @abstractmethod
    def _init_action_space(self):
        # different for discrete and continuous environments
        pass

    @staticmethod
    @abstractmethod
    def _translate_raw_action(raw_action) -> tuple[float, float]:
        # different for discrete and continuous environments
        pass

    @abstractmethod
    def _reset(self):
        """Must call self._ship_state.set()"""
        pass

    @abstractmethod
    def _reward(self) -> float:
        pass


class DiscreteSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        # engine can be turned on or off: 2 options
        # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
        self.action_space = Discrete(2 * 3)

    @staticmethod
    def _translate_raw_action(raw_action: int) -> tuple[float, float]:
        if raw_action == 0:
            return 0.0, 0.0
        elif raw_action == 1:
            return 1.0, 0.0
        elif raw_action == 2:
            return 0.0, -1.0
        elif raw_action == 3:
            return 0.0, 1.0
        elif raw_action <= 5:
            return 1.0, (raw_action - 4.5) * 2
        else:
            raise ValueError


class ContinuousSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        ones = np.ones(2, dtype=np.float32)
        self.action_space = Box(low=-ones, high=ones)

    @staticmethod
    def _translate_raw_action(raw_action: np.array) -> tuple[float, float]:
        engine_action, thruster_action = raw_action
        # [-1, 1] -> [0, 1]
        return (engine_action + 1) / 2, thruster_action
