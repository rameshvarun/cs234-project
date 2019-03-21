from baselines.common.tf_util import get_session

import pandas as pd

import stock_order
import gym
import random

import tempfile

import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.deepq.deepq import ActWrapper

from baselines.common.tf_util import get_session

from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import pywt


import matplotlib.pyplot as plt

STOCKS = ["AAPL", "MSFT", "IBM", "AMZN", "HP", "INTC", "GOOG"]

PRICE_HISTORY = 20
TRAIN_EPISODES = 50

TRAIN_START_DATE = pd.to_datetime("2008-12-31").tz_localize("US/Eastern")
TRAIN_END_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")

VALIDATE_START_DATE = pd.to_datetime("2016-12-31").tz_localize("US/Eastern")
VALIDATE_END_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")

TEST_START_DATE = pd.to_datetime("2017-12-31").tz_localize("US/Eastern")
TEST_END_DATE = pd.to_datetime("2018-12-31").tz_localize("US/Eastern")


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
def lstmnew(n_units = 32, n_features = 20):

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


def learn(
    network,
    wavelet = False,
    seed=None,
    lr=5e-4,
    total_timesteps=100_000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    train_freq=1,
    batch_size=32,
    print_freq=100,
    checkpoint_freq=10000,
    checkpoint_path=None,
    learning_starts=1000,
    gamma=1.0,
    target_network_update_freq=500,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=1e-6,
    test=False,
    **network_kwargs,
):
    """Train a deepq model.
    Parameters
    -------
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    **network_kwargs
        additional keyword arguments to pass to the network builder.
    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = stock_order.observation_space(PRICE_HISTORY)

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=stock_order.action_space().n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=None,
    )

    act_params = {
        "make_obs_ph": make_obs_ph,
        "q_func": q_func,
        "num_actions": stock_order.action_space().n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha=prioritized_replay_alpha
        )
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(
            prioritized_replay_beta_iters,
            initial_p=prioritized_replay_beta0,
            final_p=1.0,
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * total_timesteps),
        initial_p=1.0,
        final_p=exploration_final_eps,
    )

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    t = 0
    for episode in range(TRAIN_EPISODES):
        print(f"======= Train Episode {episode} =======")
        asset = random.choice(STOCKS)
        env = gym.make(
            "StockOrder-v0",
            asset=asset,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            price_history=PRICE_HISTORY,
        )

        obs = env.reset()
        done = False

        while not done:
            # Take action and update exploration to the newest value
            update_eps = exploration.value(t)
            action = act(np.array(obs)[None], update_eps=update_eps)[0]
            env_action = action

            new_obs, rew, done, _ = env.step(env_action)
            t += 1.0

            x = new_obs[0:20]
            #print('TPYE:',type(x))
            #print('X[1]:',x[1:20])
            #np.reshape(x, [20,1])
            #print('shape of x is:', x.shape)

            #for i in range(0, x.shape[1]):

            if wavelet == True:
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

                    new_obs[0:20] = temp_array[:n]


            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            if t > learning_starts and t % train_freq == 0 or done:
                if prioritized_replay:
                    experience = replay_buffer.sample(
                        batch_size, beta=beta_schedule.value(t)
                    )
                    (
                        obses_t,
                        actions,
                        rewards,
                        obses_tp1,
                        dones,
                        weights,
                        batch_idxes,
                    ) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(
                        batch_size
                    )
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

                update_target()

    # Test/Validate Environment
    portfolio, sortino, sharpe = [], [], []
    action_counts = [0, 0, 0]
    for asset in STOCKS:
        print(f"======= {'Test' if test else 'Validate'} {asset} Episode =======")

        env = gym.make(
            "StockOrder-v0",
            asset=asset,
            start_date=TEST_START_DATE if test else VALIDATE_START_DATE,
            end_date=TEST_END_DATE if test else VALIDATE_END_DATE,
            price_history=PRICE_HISTORY,
        )

        obs = env.reset()
        done = False

        while not done:
            update_eps = exploration.value(t)
            action = act(np.array(obs)[None], update_eps=update_eps)[0]
            obs, r, done, info = env.step(action)
            action_counts[action] += 1

        portfolio.append(env.perf.portfolio_value[-1])
        sortino.append(env.perf.sortino[-1])
        sharpe.append(env.perf.sharpe[-1])

        env.perf.portfolio_value.plot()

    print(f"Portfolio Value: {np.mean(portfolio)} +/- {np.std(portfolio)}")
    print(f"Sortino Ratio: {np.mean(sortino)} +/- {np.std(sortino)}")
    print(f"Sharpe Ratio: {np.mean(sharpe)} +/- {np.std(sharpe)}")

    action_counts = action_counts / np.sum(action_counts)
    print(f"Action Proportion - STAY: {action_counts[0]}, BUY: {action_counts[1]}, SELL: {action_counts[2]}")

    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value")
    plt.legend(STOCKS)
    plt.savefig("stock_order_lstm.png")

    return act


if __name__ == "__main__":
    set_global_seeds(None)
    act = learn(
        network="lstmnew",
        wavelet = True,
        lr=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
