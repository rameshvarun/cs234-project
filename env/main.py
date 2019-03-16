import stock_order
import gym
import datetime

from baselines import deepq
import pandas as pd

STOCKS = ["AAPL", "MSFT", "GOOG", "IBM", "FB", "TWTR", "AMZN", "HP", "INTC"]
TESTING_START_DATE = pd.to_datetime("2008-12-31").tz_localize("US/Eastern")
TESTING_END_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")

if __name__ == "__main__":
    env = gym.make(
        "StockOrder-v0",
        asset="AAPL",
        start_date=TESTING_START_DATE,
        end_date=TESTING_END_DATE,
        price_history=20,
    )
    print(env.action_space)
    print(env.observation_space)
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
    print("Saving model to portfolio_balance.pkl")
    act.save("cartpole_model.pkl")
