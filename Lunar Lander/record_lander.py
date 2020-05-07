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
    model_path = '/Users/gryslik/gitRepos/qlearning/test_code/lunar_lander/models_large/'
    base_path = '/Users/gryslik/gitRepos/qlearning/test_code/lunar_lander/'
else:
    model_path = '/home/ubuntu/data/code/mountaincar/models/'


################################################
# Record good model
#################################################
model_file_path = model_path + "episode-492_model_success.h5"
model = tf.keras.models.load_model(model_file_path)

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, base_path+"/video/good_model_492/")

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
env.close()

print("Episode Reward: " + str(episode_reward))


################################################
# Record medium model
#################################################
model_file_path = model_path + "episode-100_model_failure.h5"
model = tf.keras.models.load_model(model_file_path)

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, base_path+"/video/medium_model_100/")

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
env.close()

print("Episode Reward: " + str(episode_reward))

################################################
# Record bad model
#################################################
model_file_path = model_path + "episode-30_model_failure.h5"
model = tf.keras.models.load_model(model_file_path)

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, base_path+"/video/start_model_30/")

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
env.close()

print("Episode Reward: " + str(episode_reward))
