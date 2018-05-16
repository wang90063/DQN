import gym
env=gym.make('Breakout-v0')
import tensorflow as tf
import numpy as np
import  random

class testAgent():

    def __init__(self,actions):
        # init some parameters
        self.timeStep = 0
        self.epsilon = 0.05
        self.actions = actions
        # init Q network
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print ("Could not find old network weights")


    def createQNetwork(self):
        # network weights
        with tf.device('/device:GPU:0'):
           W_conv1 = self.weight_variable([8,8,4,32])
           b_conv1 = self.bias_variable([32])

           W_conv2 = self.weight_variable([4,4,32,64])
           b_conv2 = self.bias_variable([64])

           W_conv3 = self.weight_variable([3,3,64,64])
           b_conv3 = self.bias_variable([64])

           W_fc1 = self.weight_variable([3136,512])
           b_fc1 = self.bias_variable([512])

           W_fc2 = self.weight_variable([512,self.actions])
           b_fc2 = self.bias_variable([self.actions])

        # input layer

           stateInput = tf.placeholder("float",[None,84,84,4])

        # hidden layers
           h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)

           h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

           h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
           h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
           h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
           QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2


    def setPerception(self,nextObservation,action,reward,terminal):
        nextObservation = np.reshape(nextObservation,[nextObservation.shape[0],nextObservation.shape[1],1])
        newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
        self.currentState = newState
        self.timeStep += 1

    def getAction(self,done):
        QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
        action = np.zeros(self.actions)
        if done:
           action[0]=1
        else:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1

        return action

    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")