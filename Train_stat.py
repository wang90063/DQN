#---------------------------------
#
#Return the average episode reward at each epoch
#Epoch length=50000 updates


#---------------------------------

import tensorflow as tf
import numpy as np
import random


EPSILON = 0.05

class Train_stat:

    def __init__(self, actions, max_actions):

        self.timeStep = 0
        self.actions = max_actions
        self.real_actions = actions

        # init Q network
        self.stateInput, self.QValue, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNetwork_ram()

        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights in train static")




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


    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        if random.random() <= EPSILON:
            action_index = random.randrange(self.real_actions)
            action[action_index] = 1
        else:
            action_index = np.argmax(QValue[:self.real_actions])
            action[action_index] = 1

        return action


    def setPerception(self, nextObservation):
        newState = nextObservation

        print("TIMESTEP", self.timeStep, "/ STATE", 'Train_static', "/ EPSILON", EPSILON)

        self.currentState = newState
        self.timeStep += 1



    def setInitState(self, observation):
        self.currentState = observation

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)