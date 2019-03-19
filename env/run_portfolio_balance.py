import os
import time

import portfolio_balance

import gym
import datetime

import matplotlib.pyplot as plt

import pickle

import random

# from baselines.ppo1 import pposgd_simple, mlp_policy
from baselines.ddpg import ddpg
import pandas as pd

from collections import deque

import numpy as np

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

STOCKS = ["AAPL", "MSFT", "IBM", "AMZN", "HP", "INTC"]

NUM_STOCKS = 5
PRICE_HISTORY = 20

TRAIN_EPISODES = 10

TRAIN_START_DATE = pd.to_datetime("2008-12-31").tz_localize("US/Eastern")
TRAIN_END_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")

VALIDATE_START_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")
VALIDATE_END_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")

TEST_START_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")
TEST_END_DATE = pd.to_datetime("2018-12-31").tz_localize("US/Eastern")

BUDGET = 1000

from portfolio_balance import observation_space, action_space


def learn(
    network,
    test=False,
    nb_epochs=None,  # with default settings, perform 1M steps total
    nb_epoch_cycles=20,
    nb_rollout_steps=100,
    reward_scale=1.0,
    noise_type="adaptive-param_0.2",
    normalize_returns=False,
    normalize_observations=True,
    critic_l2_reg=1e-2,
    actor_lr=1e-4,
    critic_lr=1e-3,
    popart=False,
    gamma=0.99,
    clip_norm=None,
    nb_train_steps=50,  # per epoch cycle and MPI worker,
    nb_eval_steps=100,
    batch_size=64,  # per MPI worker
    tau=0.01,
    param_noise_adaption_interval=50,
    **network_kwargs,
):
    nb_actions = action_space(NUM_STOCKS).shape[-1]
    assert (
        np.abs(action_space(NUM_STOCKS).low) == action_space(NUM_STOCKS).high
    ).all()  # we assume symmetric actions.

    memory = Memory(
        limit=int(1e6),
        action_shape=action_space(NUM_STOCKS).shape,
        observation_shape=observation_space(NUM_STOCKS, PRICE_HISTORY).shape,
    )
    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    param_noise = AdaptiveParamNoiseSpec(
        initial_stddev=float(0.2), desired_action_stddev=float(0.2)
    )

    max_action = action_space(NUM_STOCKS).high
    print("scaling actions by {} before executing in env".format(max_action))

    agent = DDPG(
        actor,
        critic,
        memory,
        observation_space(NUM_STOCKS, PRICE_HISTORY).shape,
        action_space(NUM_STOCKS).shape,
        gamma=gamma,
        tau=tau,
        normalize_returns=normalize_returns,
        normalize_observations=normalize_observations,
        batch_size=batch_size,
        action_noise=None,
        param_noise=param_noise,
        critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        enable_popart=popart,
        clip_norm=clip_norm,
        reward_scale=reward_scale,
    )
    print("Using agent with the following configuration:")
    print(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()

    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()

    for episode in range(TRAIN_EPISODES):
        print(f"======= Train Episode {episode} =======")
        assets = random.sample(STOCKS, NUM_STOCKS)
        env = gym.make(
            "PortfolioBalance-v0",
            assets=assets,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            budget=BUDGET,
            price_history=PRICE_HISTORY,
        )

        obs = env.reset()
        done = False

        while not done:
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, _, _ = agent.step(
                    np.array([obs]), apply_noise=True, compute_Q=True
                )
                new_obs, r, done, info = env.step(max_action * action)

                # Book-keeping.
                agent.store_transition(
                    np.array([obs]),
                    np.array([action]),
                    np.array([r]),
                    np.array([new_obs]),
                    np.array([done]),
                )  # the batched data will be unrolled in memory.py's append.

                obs = new_obs

                if done:
                    break

            agent.reset()

            # Train.
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if (
                    memory.nb_entries >= batch_size
                    and t_train % param_noise_adaption_interval == 0
                ):
                    agent.adapt_param_noise()

                cl, al = agent.train()
                agent.update_target_net()

    # Test/Validate Environment
    print(f"======= {'Test' if test else 'Validate'} Episode =======")
    assets = random.sample(STOCKS, NUM_STOCKS)
    env = gym.make(
        "PortfolioBalance-v0",
        assets=assets,
        start_date=TEST_START_DATE if test else VALIDATE_START_DATE,
        end_date=TEST_END_DATE if test else VALIDATE_END_DATE,
        budget=BUDGET,
        price_history=PRICE_HISTORY,
    )

    obs = env.reset()
    done = False

    while not done:
        action, q, _, _ = agent.step(np.array([obs]), apply_noise=True, compute_Q=True)
        obs, r, done, info = env.step(max_action * action)

    env.perf.portfolio_value.plot()
    plt.title(f"PortfolioBalanceEnvironment Portfolio Value ({'Test' if test else 'Validation'})")
    plt.savefig("portfolio_return.png")

    return agent


if __name__ == "__main__":
    set_global_seeds(None)
    act = learn(network="mlp", test=False)
