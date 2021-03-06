import random
import numpy as np
import os
import tensorflow as tf
import gym
import collections
import pandas as pd
import getpass
import time

current_user = getpass.getuser()
if(current_user == "gryslik"):
    model_path = '/Users/gryslik/gitRepos/qlearning/test_code/lunar_lander/models_large_doubleDQN/'
    #model_path = '/Users/gryslik/gitRepos/qlearning/test_code/lunar_lander/models_large/'
    base_path = '/Users/gryslik/gitRepos/qlearning/test_code/lunar_lander/'
else:
    model_path = '/home/ubuntu/data/code/mountaincar/models/'


model_file_path = model_path + "episode-493_model_success.h5" #490 gives about 250, some give poorly -- in models large?
model = tf.keras.models.load_model(model_file_path)

env = gym.make('LunarLander-v2')
#env = gym.wrappers.Monitor(env, base_path+"/video/good_model_193/")

step_list = []
reward_list = []
for i in range(100):
    print("============================================")
    print("Processing Episode: " + str(i))
    print("============================================")
    time_start = time.time()
    state = env.reset().reshape(1, 8)
    episode_reward = 0
    step = 0
    done = False
    while not done:  # will auto terminate when it reaches 200
        prediction_values = np.array(model.predict_on_batch(state))
        action = np.argmax(prediction_values)
        state, reward, done, info = env.step(action)
        state = state.reshape(1, 8)
        step += 1
        episode_reward += reward
        #env.render()
    step_list.append(step)
    reward_list.append(episode_reward)
    print(episode_reward)
    time_end = time.time()
    print("Processed in: " + str(int(time_end-time_start)))
env.close()

np.array(reward_list).mean()