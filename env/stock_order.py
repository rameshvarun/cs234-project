import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register

import zipline
import threading

import numpy as np

from queue import Queue


def zipline_thread(
    start_date,
    end_date,
    asset,
    price_history,
    actions_queue,
    observations_queue,
    returns_queue,
):
    print("Starting Zipline thread...")

    def zipline_initialize(context):
        context.previous_value = None

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
            zipline.api.order_target(zipline.api.symbol(asset), 100)
        elif action == 2:
            zipline.api.order_target(zipline.api.symbol(asset), 0)
        else:
            raise f"Unknown action {action}"

    zipline.run_algorithm(
        start=start_date,
        end=end_date,
        initialize=zipline_initialize,
        capital_base=10000000.0,
        handle_data=zipline_handle_data,
        bundle="quandl",
    )
    print("Zipline thread finished...")
    observations_queue.put(None)


LOWEST_ASSET_PRICE = 0
HIGHEST_ASSET_PRICE = 200000


class StockOrder(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, assets, start_end_date_train, start_end_date_validate, start_end_date_test, price_history):
        self.observation_space = spaces.Box(
            low=np.array([LOWEST_ASSET_PRICE] * price_history),
            high=np.array([HIGHEST_ASSET_PRICE] * price_history),
        )
        self.action_space = spaces.Discrete(3)
        self.start_date_train, self.end_date_train = start_end_date_train
        self.start_date_validate, self.end_date_validate = start_end_date_validate
        self.start_date_test, self.end_date_test = start_end_date_test
        self.assets = assets
        self.asset = self.assets[0]
        # # self.asset = self.assets.pop()
        self.last_asset = False
        self.price_history = price_history

    def reset(self):
        print("Resetting environment...")
        self.actions_queue = Queue()
        self.observations_queue = Queue()
        self.returns_queue = Queue()
        if not self.assets:
            self.last_asset = True
        else:
            self.asset = self.assets.pop()
        print(f"Now training using {self.asset} data.")
        self.sim_thread = threading.Thread(
            target=zipline_thread,
            kwargs={
                "start_date": self.start_date_train,
                "end_date": self.end_date_train,
                "asset": self.asset,
                "price_history": self.price_history,
                "actions_queue": self.actions_queue,
                "observations_queue": self.observations_queue,
                "returns_queue": self.returns_queue,
            },
        )
        self.sim_thread.start()
        observation = self.observations_queue.get()
        return observation#, False
        # if self.assets:
        #     self.asset = self.assets.pop()
        #     print(f"Now training using {self.asset} data.")
        #     self.sim_thread = threading.Thread(
        #         target=zipline_thread,
        #         kwargs={
        #             "start_date": self.start_date_train,
        #             "end_date": self.end_date_train,
        #             "asset": self.asset,
        #             "price_history": self.price_history,
        #             "actions_queue": self.actions_queue,
        #             "observations_queue": self.observations_queue,
        #             "returns_queue": self.returns_queue,
        #         },
        #     )
        #     self.sim_thread.start()
        #     observation = self.observations_queue.get()
        #     return observation, False
        # else:
        #     return 0, True

    def step(self, action):
        self.actions_queue.put(action)

        observation = self.observations_queue.get()
        if observation is None:
            return np.zeros(shape=(self.price_history,)), 0, True, {}
        else:
            reward = self.returns_queue.get()
            return observation, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


register(id="StockOrder-v0", entry_point="stock_order:StockOrder")
