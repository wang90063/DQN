
import tensorflow as tf
import numpy as np
import random
import gym
env=gym.make('Qbert-ram-v0')

action_max_dim = 18  # Largest action dimension#
action_dim = env.action_space.n

EPSILON = 0.05

# Build the network

with tf.device('/device:GPU:0'):
    W_fc1 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.01, shape=[64]))

    W_fc2 = tf.Variable(tf.truncated_normal([64,32], stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.01, shape=[32]))

    W_fc3 =  tf.Variable(tf.truncated_normal([32,action_max_dim], stddev=0.01))
    b_fc3 = tf.Variable(tf.constant(0.01, shape=[action_max_dim]))

    # input layer

    stateInput = tf.placeholder("float", [None, 128])

    # hidden layers

    h1 = tf.nn.relu(tf.matmul(stateInput, W_fc1) + b_fc1)

    h2 = tf.nn.relu(tf.matmul(h1, W_fc2) + b_fc2)

    # output layer
    QValue = tf.matmul(h2, W_fc3) + b_fc3

saver = tf.train.Saver()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
checkpoint_list = tf.train.get_checkpoint_state("saved_networks").all_model_checkpoint_paths


train_stat = []


for i in range(len(checkpoint_list)):

    saver.restore(session, checkpoint_list[i])

    Timestep = 0


    # Total reward for an episode
    episode_reward_stat = 0
    # The total time steps in each episode
    step_episode_stat = 1
    # Average episode reward in each episode
    episode_reward_stat_total = 0
    # Episode number
    episode_num_stat=-1
    done_stat=True
    while Timestep<10000:
        if done_stat:
            episode_reward_stat_total+= episode_reward_stat/step_episode_stat
            episode_reward_stat = 0
            step_episode_stat = 0
            episode_num_stat+=1
            observation_stat = env.reset()
            current_state = observation_stat

        #Get action

        Q=QValue.eval(feed_dict={stateInput: [current_state]})[0]
        action = np.zeros(action_max_dim)
        if random.random() <= EPSILON:
            action_index = random.randrange(action_dim)
            action[action_index] = 1
        else:
            action_index = np.argmax(QValue[:action_dim])
            action[action_index] = 1

        nextObservation_stat, reward_stat, done_stat, info_stat = env.step(np.argmax(action))

        episode_reward_stat += reward_stat
        step_episode_stat += 1

        # Update the state

        current_state=nextObservation_stat

        Timestep+=1

        print("EPOCH", i, "/ TIMESTEP", Timestep)

    train_stat.append(episode_reward_stat_total/episode_num_stat)


with open("data/Qbert.txt", 'a') as k:
    for ts in train_stat:
        k.write(str(ts) + '\n')
    k.close()



