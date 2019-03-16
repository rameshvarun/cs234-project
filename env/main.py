import stock_order
import gym
import datetime

from baselines import deepq
import pandas as pd

STOCKS = ["AAPL", "MSFT", "GOOG", "IBM", "FB", "TWTR", "AMZN", "HP", "INTC"]

TRAINING_START_DATE = pd.to_datetime("2008-12-31").tz_localize("US/Eastern")
TRAINING_END_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")

VALIDATE_START_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")
VALIDATE_END_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")

TEST_START_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")
TEST_END_DATE = pd.to_datetime("2018-12-31").tz_localize("US/Eastern")

if __name__ == "__main__":
    env = gym.make(
        "StockOrder-v0",
        asset="AAPL",
        start_date=TRAINING_START_DATE,
        end_date=TRAINING_END_DATE,
        price_history=20,
    )
    act = deepq.learn(
        env,
        network="mlp",
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    print("Saving model to stock_order.pkl")
    act.save("stock_order.pkl")
