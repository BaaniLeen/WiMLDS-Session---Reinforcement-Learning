import gym
import gym_maze
import numpy as np
from copy import deepcopy
from math import floor
import pickle
import random

NO_OF_TIME_LIMIT=2500
NO_OF_EPISODES=50
env = gym.make("maze-sample-5x5-v0")

env.reset()
env.render()

learning_rate = 0.8
gamma=0.9 ## discount factor
convergence_counts=[]
time_count=[]
qTable = np.zeros((5,5)+(4,), dtype= float)
for j in range(NO_OF_EPISODES):
    print("Episode ",j)
    if(len(time_count)>5 and time_count[:-1]==time_count[:-2] and time_count[:-1]==time_count[:-3] and time_count[:-1]==time_count[:-4] and time_count[:-1]==time_count[:-5]):
        convergence_counts.append(j-5)
        break
    epsilon=1
    env.reset()
    observation=[]
    observation.append(0)
    observation.append(0)
    total_reward=0
    for i in range(NO_OF_TIME_LIMIT):
        # print("Time: ",i)
        prev_state = tuple(deepcopy(observation))
        # print("Prev:  "+str(prev_state))       
        # action = env.action_space.sample()
        a=random.random()
        if(epsilon<a):
            action = env.action_space.sample()
        else:
            action = int(np.argmax(qTable[prev_state]))
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = tuple(deepcopy(observation))
        # print("Prev:  "+str(prev_state))
        # print("Curr: "+str(current_state))
        best_q = np.amax(qTable[current_state]) 
        # print(qTable)
        # print(best_q)
        qTable[prev_state + (action,)] += learning_rate * (reward + gamma * (best_q) - qTable[prev_state + (action,)])
        total_reward+=reward
        # print(qTable)
        if done:
            # print(str(j)+" "+str(i)+ " "+ str(total_reward)+"\n")
            time_count.append(i)
            print("Time: ",i)
            break
    # print(str(learning_rate)+" "+str(convergence_counts)+"\n")