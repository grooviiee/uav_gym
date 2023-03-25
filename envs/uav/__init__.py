import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace

logger = logging.getLogger(__name__)


class UavGym(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        full_observable: bool = False,
        step_cost: float = 0,
        n_agent: int = 4,
        max_steps: int = 50,
        clock: bool = True,
    ):

        self._grid_shape = (10000, 10000)
        self.n_agents = n_agent
        self.max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self._total_episode_reward = None
        self._add_clock = clock
        self._agent_dones = None

        self.action_spzce = MultiAgentActionSpace(
            [spaces.Discrete(5) for _ in range(self.n_agents)]
        )
