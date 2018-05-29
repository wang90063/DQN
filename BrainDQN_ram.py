# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.95  # decay rate of past observations
OBSERVE = 50000.  # timesteps to observe before training
EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.1  # 0.001 # final value of epsilon
INITIAL_EPSILON = 1.0  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 1000000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 10000


class BrainDQN_ram:
    def __init__(self, actions, max_actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = max_actions
        self.real_actions = actions
        self.epsilon = INITIAL_EPSILON
        # init Q network
        self.stateInput, self.QValue, self. W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNetwork_ram()
        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T, self.W_fc3T, self.b_fc3T = self.createQNetwork_ram()

        self.copyTargetQNetworkOperation = \
            [self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1), self.W_fc2T.assign(self.W_fc2),
             self.b_fc2T.assign(self.b_fc2), self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def createQNetwork_ram(self):

        with tf.device('/device:GPU:0'):
            W_fc1 = self.weight_variable([128, 64])
            b_fc1 = self.bias_variable([ 64])

            W_fc2 = self.weight_variable([64, 32])
            b_fc2 = self.bias_variable([32])

            W_fc3 = self.weight_variable([32, self.actions])
            b_fc3 = self.bias_variable([ self.actions])

            # input layer

            stateInput = tf.placeholder("float", [None, 128])

            # hidden layers

            h1 = tf.nn.relu(tf.matmul(stateInput, W_fc1) + b_fc1)

            h2 = tf.nn.relu(tf.matmul(h1, W_fc2) + b_fc2)

            # output layer
            QValue = tf.matmul(h2, W_fc3) + b_fc3

        return stateInput, QValue, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 50000==0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = nextObservation
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.real_actions)
            action[action_index] = 1
        else:
            action_index = np.argmax(QValue[:self.real_actions])
            action[action_index] = 1

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = observation

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

