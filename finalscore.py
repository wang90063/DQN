import gym
env=gym.make('Freeway-ram-v0')
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib as plt
from testAgent import *

def preprocess(observation):
    processed_observation = np.uint8(
        resize(rgb2gray(observation), (84, 84), mode='constant') * 255)

    return processed_observation



def testplay():
    # Step 1: init BrainDQN

    action_dim=env.action_space.n
    brain = testAgent(action_dim)

    # Init the state at the very beginning
    done = True

    # Step 2: Interaction and Training

    #Total reward for an episode
    episode_reward=0
    #The episode number
    episode_num = 1
    #The total time steps in each episode
    step_episode=1
    # maximum no operation time steps at episode beginning
    no_op_max=30

    #Average episode reward in each episode
    episode_reward_list=[]

    while episode_num<30:
        env.render()
        if done:

            episode_reward_list.append(episode_reward)
            episode_reward=0
            step_episode = 0


            observation = env.reset()
            observation_gray = preprocess(observation) # delete when env is in ram mode
            # brain.setInitState(observation)
            brain.setInitState(observation_gray)

        action = brain.getAction(done)
        nextObservation,reward,done,info = env.step(np.argmax(action))
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,done)

        episode_reward += reward
        step_episode += 1

    # Plot the "average episode reward" v.s. "episode number"
    plt.plot(np.arange(len(episode_reward_list)), episode_reward_list)
    plt.ylabel('Average Episode Reward')
    plt.xlabel('Episode')
    plt.show()

def main():
    testplay()

if __name__ == '__main__':
    main()





