#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:18:53 2017

@author: drlego
"""

# Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

# Number of episodes to run
# 50000 episodes take over 1 week on single GPU
EPISODES = 50000

# Define DQN Agent
class DQNAgent(object):
    """DQN Agent class"""
    def __init__(self, action_size):
        # Configurations
        self.render = True
        self.load_model = True
        # Environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # Epsilon parameters
        self.epsilon = 0.3 			         # original paper; 1.0
        self.epsilon_start = 0.3 		      # original paper; 1.0
        self.epsilon_end = 0.1
        self.exploration_steps = 1000. 		# original paper; 1000000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                   / self.exploration_steps
        # Training parameters
        self.batch_size = 32
        self.train_start = 10000 		      # original paper; 50000
        self.update_target_rate = 5000 		# original paper; 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # Build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimize = self.optimize()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max = 0.
        self.avg_loss = 0.
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(logdir='summary/breakout_dqn',
                                                    graph=self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights('./save_model/breakout_dqn.h5')

    # If the error is in [-1, 1], then the cost is quadratic to the error
    # But if it is outside the interval, the cost is linear to the error
    def optimize(self):
        # a denotes action, y denotes prediction of model
        a = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # Approximate Q function using CNN
    # Input: state
    # Output: action
    def build_model(self):
        """
        Approximate Q function using CNN.
        """
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # After some time interval, update the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # Save sample <s,a,r,s'> to the replay memory
    def save_to_replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append([history, action, reward, next_history, dead])

    # Pick samples randomly from replay memory (with batch_size)
    def train_with_replay_memory(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # From replay memory, sample a mini-batch
        # one unit in memory = [history, action, reward, next_history, dead]
        mini_batch = random.sample(self.memory, self.batch_size)

        # Define shape of history & next history
        history_shape = (self.batch_size, ) + self.state_size
        history = np.zeros(shape=history_shape)
        next_history = np.zeros(shape=history_shape)

        target = np.zeros((self.batch_size, ))

        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        # Like Q-learning, set maximum Q-value at s' as target value
        # by predicting it with the target network! (model-free rl)
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i] + self.discount_factor * 0.
            else:
                target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimize([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, filename):
        self.model.save_weights(filename)

    # Make summary operators for Tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(initial_value=0.)
        episode_avg_max_q = tf.Variable(initial_value=0.)
        episode_duration = tf.Variable(initial_value=0.)
        episode_avg_loss = tf.Variable(initial_value=0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(dtype=tf.float32) for _ in
                                range(len(summary_vars))]
        assert len(summary_vars) == len(summary_placeholders)

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op


# 210 x 160 x 3 (color) --> 84 x 84 (gray)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # Do nothing at the start of episodes to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        # At start of episodes, there is no preceding frame
        # Just copy initial states to make history
        assert observe.shape == (210, 160, 3)

        state = pre_processing(observe)
        assert state.shape == (84, 84)

        history = np.stack([state, state, state, state], axis=2)
        assert history.shape == (84, 84, 4)

        history = np.reshape(history, (1, 84, 84, 4))
        assert history.shape == (1, 84, 84, 4)

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # Get action for the current history and go one step in the environment
            action = agent.get_action(history)
            # Change action to real action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            elif action == 2:
                real_action = 3

            observe, reward, done, info = env.step(real_action)
            assert observe.shape == (210, 160, 3)
            # Preprocess the observation --> history
            next_state = pre_processing(observe)
            assert next_state.shape == (84, 84)

            next_state = np.reshape([next_state], (1, 84, 84, 1))
            assert next_state.shape == (1, 84, 84, 1)

            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            # If the agent missed the ball, agent is dead. but episode is not over.
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            # Save the sample <s,a,r,s'> to the replay memory
            agent.save_to_replay_memory(history, action, reward, next_history, dead)
            # Train model
            agent.train_with_replay_memory()
            # Update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()
                print('Updating target model!')

            score += reward

            # If the agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # If episode is done, plot the scores
            if done:
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i],
                                       feed_dict={agent.summary_placeholders[i]: float(stats[i])
                                                  }
                                       )
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print(' episode:', e, ' score:', score, ' memory_length:', len(agent.memory),
                      ' epsilon:', agent.epsilon, ' global_step:', global_step,
                      ' average_q:', agent.avg_q_max / float(step), ' average_loss:', agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0., 0.

        if (e + 1) % 100 == 0:
            agent.model.save_weights('./save_model/breakout_dqn_episode_{}.h5'.format(e + 1))