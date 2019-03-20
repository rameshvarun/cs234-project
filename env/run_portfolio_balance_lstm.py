import time
import portfolio_balance
import pywt

import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn

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


mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


@register("lstmnew")
def lstmnew(n_units = 5, n_features = 20):
    
    def network_fn(X):
        #layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}
        #print('SHAPE:', X.shape, X.shape[1], int(X.get_shape().as_list()[1]/5))
        n_features = int(X.get_shape().as_list()[1]/5)
        x = tf.split(X, n_features, 1)
        #print('Shape of X:', X.shape)
        #print('Shape of x:', x[1])

        lstm_cell = rnn.BasicLSTMCell(n_units)

        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        
        output = outputs[-1]

        #output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

        return output
    
    return network_fn

def get_network_builder(name):

    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))

def build_q_func(network, hiddens=[10], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        #from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder

    
    
    
    
    

class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network_builder(obs)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

    

def learn(
    network,
    wavelet = False,
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
    critic = Critic(network='mlp', **network_kwargs)
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


                if wavelet == True:
                    for i in range(5):
                        x = new_obs[20*(i-1):20*i]

                        n = x.size
                        wavelet_transform_iterations = 1
                        for j in range(0, wavelet_transform_iterations):
                            coefficients = pywt.wavedec(x, 'db2', mode='symmetric', level=None, axis=0)
                            coefficients_transformed = []
                            coefficients_transformed.append(coefficients[0])
                            for detail_coefficient in coefficients[1:]:
                                coefficients_transformed.append(
                                    pywt.threshold(detail_coefficient, np.std(detail_coefficient), mode='garrote'))

                            temp_array = pywt.waverec(coefficients_transformed, 'db2', mode='symmetric', axis=0)

                            new_obs[20*(i-1):20*i] = temp_array[:n]

                new_obs = new_obs.reshape([5,20])
                new_obs = new_obs.transpose()
                new_obs = new_obs.reshape([1,-1])
                
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
    act = learn(network="lstmnew", wavelet = True, test=False)