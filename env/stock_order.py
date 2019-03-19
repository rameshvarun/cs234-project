import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register

import zipline
import threading

import numpy as np

from queue import Queue
from zipline.finance import commission, slippage


def zipline_thread(
    start_date,
    end_date,
    asset,
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

        history = data.history(
            zipline.api.symbol(asset), "price", bar_count=price_history, frequency="1d"
        )
        observations_queue.put(history)

        action = actions_queue.get()

        if action == 0:
            pass
        elif action == 1:
            zipline.api.order(zipline.api.symbol(asset), 10)
        elif action == 2:
            zipline.api.order(zipline.api.symbol(asset), -10)
        else:
            raise ValueError(f"Unknown action {action}")

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


LOWEST_ASSET_PRICE = 0
HIGHEST_ASSET_PRICE = 200_000


def observation_space(price_history):
    return spaces.Box(
        low=np.array([LOWEST_ASSET_PRICE] * price_history),
        high=np.array([HIGHEST_ASSET_PRICE] * price_history),
    )


def action_space():
    return spaces.Discrete(3)


class StockOrder(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, asset, start_date, end_date, price_history):
        self.observation_space = observation_space(price_history)
        self.action_space = action_space()

        self.start_date = start_date
        self.end_date = end_date
        self.asset = asset
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
                "asset": self.asset,
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
            return np.zeros(shape=(self.price_history,)), 0, True, {}
        else:
            reward = self.returns_queue.get()
            return observation, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


register(id="StockOrder-v0", entry_point="stock_order:StockOrder")
