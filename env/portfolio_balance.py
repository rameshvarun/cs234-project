import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register

import zipline


class PortfolioBalance(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, stocks, start_date, end_date):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass


register(id="PortfolioBalance-v0", entry_point="portfolio_balance:PortfolioBalance")
