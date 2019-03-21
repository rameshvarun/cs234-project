import quandl
import tensorflow as tf
import numpy  as np
import sys
from lib2 import plotting
import psutil
import itertools
import numpy as np
import os
from collections import deque, namedtuple
import random
import pandas as pd


df_rate=pd.read_csv('monthly_csv.csv', sep=',')
df_rate["date"] = df_rate["date"].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d').strftime("%Y-%m"))
### Set Quandl key
quandl.ApiConfig.api_key = 'z4iuvThvAxQCPe9xPndR'
### Download Apple data from Quandl
stocks = ['AAPL','MSFT', "GOOG","IBM","FB","TWTR","AMZN","HP","INTC"]
stock_data = quandl.get_table('WIKI/PRICES', ticker = stocks, 
                        qopts = { 'columns': ['ticker', 'close','volume', 'date'] }, 
                        date = { 'gte': '2008-12-31', 'lte': '2016-12-31' })
input_days = 20
total_days = len(stock_data)
stock_data["day"] = stock_data["date"].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d').strftime("%d"))
stock_data["date"] = stock_data["date"].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d').strftime("%Y-%m"))
new_df = stock_data.set_index('date').join(df_rate.set_index('date')).reset_index()
new_df["date"] = new_df["date"]+"-"+new_df["day"]
new_df = new_df.sort_values(by=['ticker',"date"])
### We build a model to use the closing price and volumn of previous input_days number of days, to study
### the Q function.
# input_days = 20
# total_days = len(apple_data)
data_size = total_days - input_days
# data_size = round(data_size/10) #for dev only
in_data = np.zeros((data_size, input_days, 2))
other_data = np.zeros((data_size))
for i in range(data_size):
    for j in range(input_days):
        in_data[i][j] = new_df.iloc(0)[i+j][["close","volume"]]
    other_data[i] = new_df.iloc(0)[i]["rate"]/30.
### Based on apple_data and input_days, we construct a tensor containing data_size number of data points, 
### where each data point is the closing price and volumn of input_days consecutive days.
### Later we use the new matrix 'data' to train the model.

# data_size = total_days - input_days
# data = np.zeros((data_size, input_days, 2, 1))
# for i in range(data_size):
#     for j in range(input_days):
#         data[i][j][0][0] = apple_data[i+j][4]
#         data[i][j][1][0] = apple_data[i+j][5]
# data_size = total_days - input_days
# data = np.zeros((data_size, input_days, 2, 1))
# for i in range(data_size):
#     for j in range(input_days):
#         data[i][j][0][0] = stock_data.iloc(0)[i+j][["close","volume"]][0]
#         data[i][j][1][0] = stock_data.iloc(0)[i+j][["close","volume"]][0]
# VALID_ACTIONS = [-1, 0, 1]

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """
    
    ### summaries_dir is the address to store summaries.

    def __init__(self, scope="estimator", summaries_dir='.'):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # our input are prices and volumns of previous input_days number of days.
        self.X_pl = tf.placeholder(shape=[None, input_days, 2], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl)
        batch_size = tf.shape(self.X_pl)[0]
        
    
        #One convolutional layer
		#conv1 = tf.contrib.layers.conv2d(X, 10, 1, 1, activation_fn=tf.nn.relu)
    
        # Fully connected layers
        flattened = tf.contrib.layers.flatten(X)
        #fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        # fc2 = tf.contrib.layers.fully_connected(fc1, 512)
        # fc3 = tf.contrib.layers.fully_connected(fc2, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only   ########STOP HERE#########
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])
        
    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          IGNORE: s: State input of shape [batch_size, 4, 160, 160, 3]
          s: the price and volumns of last 20 days.

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """        
        
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """
    
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.  
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(data, other_data,
                    sess,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500,
                    replay_memory_init_size=50,
                    update_target_estimator_every=100,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500,
                    batch_size=32,
                    max_step=500
                    ):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        data: the price and volumn data
        sess: Tensorflow Session object
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        max_step: the max step in each episode
        
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

    # The replay memory
    replay_memory = []
    
    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
        
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = data[0]
    
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        #next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = data[i+1]
        reward = R(data[i][-1][0],data[i+1][0][0], VALID_ACTIONS[action], other_data[i])     
        replay_memory.append(Transition(state, action, reward, next_state))
        state = next_state
    mark = i
    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        state = data[mark+1]
        loss = None

        # One step in the environment
        for t in range(max_step):
            t += mark
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            next_state = data[i+t+1]
            reward = R(data[t][-1][0],data[t+1][0][0], VALID_ACTIONS[action],other_data[t])     
            #next_state, reward = data[t + 1], (data[t+1][0][0] - data[t][0][0])[0] * VALID_ACTIONS[action]
            #next_state = state_processor.process(sess, next_state)
            #next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state))   

            # Update statistics
            #print('reward:',reward)
            stats.episode_rewards[i_episode] += reward
            
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))
            reward_batch = reward_batch.reshape((batch_size,))
            #print('reward_batch', reward_batch.shape)
            #print('action_batch', action_batch.shape)
            #print('states_batch', states_batch.shape)



            # Calculate q values and targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + discount_factor * np.amax(q_values_next, axis=1)
            #print('target_batch', targets_batch.shape)
            #print('q_values_next', (discount_factor * np.amax(q_values_next, axis=1)).shape)
            
            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            state = next_state
            total_t += 1
            
        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
        episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_writer.flush()
        
        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
#experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
experiment_dir = os.path.abspath("./experiments")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# data = np.random.randn(1000,20,2,1)

# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(in_data, data_other,
                                    sess,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=100,
                                    replay_memory_size=500,
                                    replay_memory_init_size=50,
                                    update_target_estimator_every=10,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=50,
                                    discount_factor=0.99,
                                    batch_size=32,
                                    max_step=20
                                    ):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
