import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register

import zipline
import threading

import numpy as np

from queue import Queue
from zipline.finance import commission, slippage

LOWEST_ASSET_PRICE = 0
HIGHEST_ASSET_PRICE = 200_000
ASSET_ALLOCATION_LEVELS = 5


def zipline_thread(
    start_date,
    end_date,
    assets,
    budget,
    price_history,
    actions_queue,
    observations_queue,
    returns_queue,
    perf_queue,
):
    print("Starting Zipline thread...")

    def zipline_initialize(context):
        context.previous_value = None
        context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
        context.set_slippage(slippage.VolumeShareSlippage())

    def zipline_handle_data(context, data):
        if context.previous_value != None:
            ret = context.portfolio.portfolio_value - context.previous_value
            returns_queue.put(ret)

        context.previous_value = context.portfolio.portfolio_value

        history = np.concatenate(
            [
                data.history(
                    zipline.api.symbol(asset),
                    "price",
                    bar_count=price_history,
                    frequency="1d",
                )
                for asset in assets
            ]
        )
        observations_queue.put(history)

        action = actions_queue.get()
        action = action + 0.5
        proportion = action / (action.sum())
        values = proportion * budget

        for asset, value in zip(assets, values[0]):
            zipline.api.order_target_value(zipline.api.symbol(asset), value)

    def zipline_analyze(context, perf):
        perf_queue.put(perf)

    zipline.run_algorithm(
        start=start_date,
        end=end_date,
        initialize=zipline_initialize,
        capital_base=10_000_000.0,
        handle_data=zipline_handle_data,
        bundle="quandl",
        analyze=zipline_analyze,
    )
    print("Zipline thread finished...")
    observations_queue.put(None)


def observation_space(num_assets, price_history):
    return spaces.Box(
        low=np.full((price_history * num_assets), LOWEST_ASSET_PRICE),
        high=np.full((price_history * num_assets), HIGHEST_ASSET_PRICE),
    )


def action_space(num_assets):
    return spaces.Box(low=np.full(num_assets, -0.5), high=np.full(num_assets, 0.5))


class PortfolioBalance(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, assets, budget, start_date, end_date, price_history):
        self.observation_space = observation_space(len(assets), price_history)
        self.action_space = action_space(len(assets))

        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets
        self.budget = budget
        self.price_history = price_history

    def reset(self):
        print("Resetting environment...")
        self.actions_queue = Queue()
        self.observations_queue = Queue()
        self.returns_queue = Queue()
        self.perf_queue = Queue()

        self.sim_thread = threading.Thread(
            target=zipline_thread,
            kwargs={
                "start_date": self.start_date,
                "end_date": self.end_date,
                "assets": self.assets,
                "budget": self.budget,
                "price_history": self.price_history,
                "actions_queue": self.actions_queue,
                "observations_queue": self.observations_queue,
                "returns_queue": self.returns_queue,
                "perf_queue": self.perf_queue,
            },
        )
        self.sim_thread.start()

        observation = self.observations_queue.get()
        return observation

    def step(self, action):
        self.actions_queue.put(action)

        observation = self.observations_queue.get()
        if observation is None:
            self.perf = self.perf_queue.get()
            return (
                np.zeros(shape=(self.price_history * len(self.assets))),
                0.0,
                True,
                {},
            )
        else:
            reward = self.returns_queue.get()
            return observation, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


register(id="PortfolioBalance-v0", entry_point="portfolio_balance:PortfolioBalance")
