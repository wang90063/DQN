import gym
env=gym.make('Qbert-ram-v0')
# from BrainDQN_Nature import *
from  BrainDQN_ram import *
import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess(observation):
    processed_observation = np.uint8(
        resize(rgb2gray(observation), (84, 84), mode='constant') * 255)

    return processed_observation

def playAtari():
    # Step 1: init BrainDQN

    action_max_dim=18    # Largest action dimension#
    action_dim = env.action_space.n
    brain = BrainDQN_ram(action_dim,action_max_dim)

    # Init the state at the very beginning
    done = True

    # Step 2: Interaction and Training

    #Total reward for an episode
    episode_reward=0
    #The episode number
    episode_num = 1
    #The total time steps in each episode
    step_episode=1

    #Average episode reward in each episode
    episode_reward_list=[]

    while brain.timeStep<3000000:
        env.render()
        if done:

            episode_reward_list.append(episode_reward/step_episode)
            episode_reward=0
            step_episode = 0

            observation = env.reset()
            # observation_gray = preprocess(observation)#delete when env is in ram mode
            # brain.setInitState(observation_gray)
            brain.setInitState(observation)
        action = brain.getAction()
        nextObservation,reward,done,info = env.step(np.argmax(action))
        # nextObservation = preprocess(nextObservation)
        if reward>0:
            reward=1
        elif reward<0:
            reward=-1
        brain.setPerception(nextObservation,action,reward,done)

        episode_reward += reward
        step_episode += 1

    # Plot the "average episode reward" v.s. "episode number"
    plt.plot(np.arange(len(episode_reward_list)), episode_reward_list)
    plt.ylabel('Average Episode Reward')
    plt.xlabel('Episode')
    plt.show()
def main():
    playAtari()

if __name__ == '__main__':
    main()