import stock_order
import portfolio_balance

import gym
import datetime

# from baselines.ppo1 import pposgd_simple, mlp_policy
from baselines.ddpg import ddpg
import pandas as pd

STOCKS = ["AAPL", "MSFT", "IBM", "AMZN", "HP", "INTC"]

TRAINING_START_DATE = pd.to_datetime("2008-12-31").tz_localize("US/Eastern")
TRAINING_END_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")

VALIDATE_START_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")
VALIDATE_END_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")

TEST_START_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")
TEST_END_DATE = pd.to_datetime("2018-12-31").tz_localize("US/Eastern")

if __name__ == "__main__":
    env = gym.make(
        "PortfolioBalance-v0",
        assets=STOCKS,
        start_date=TRAINING_START_DATE,
        end_date=TRAINING_END_DATE,
        budget=1000,
        price_history=20,
    )

    act = ddpg.learn(env=env, network="mlp")
    print("Saving model to stock_order.pkl")
    act.save("stock_order.pkl")
