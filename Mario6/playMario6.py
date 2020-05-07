import random
import numpy as np
import os
import tensorflow as tf
import gym
import collections
import getpass
import time
import pandas as pd
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import warnings
sys.path.append("/Users/gryslik/gitRepos/qlearning/test_code/Mario/")
from helper_file import *


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

warnings.simplefilter("ignore", lineno=148)


current_user = getpass.getuser()
if (current_user == "gryslik"):
    model_path = '/Users/gryslik/gitRepos/qlearning/test_code/mario/models6-DDQN/'
else:
    model_path = '/home/ubuntu/data/code/mario/model6-DDQN/'


env = gym_super_mario_bros.make('SuperMarioBros-2-2-v0')   #2-2, 1-1
env = JoypadSpace(env, RIGHT_ONLY)

model_file_path = model_path + 'episode-1340_model_failure.h5' #50 vs 990   #930 shaping up #1870
model = tf.keras.models.load_model(model_file_path)

# env.action_space.sample() = numbers, for example, 0,1,2,3...
# state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
# reward = int; for example, 0, 1 ,2, ...
# done = False or True
# info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}

stack_size = 4

long_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], stack_size, axis=2) #reshape it (120x128x4).
done = False;
step_counter = 0
while not done and step_counter < 1000: # Now we need to take the same action every 4 steps
    prediction_values = model.predict(np.expand_dims(long_state, axis=0).astype('float16'))
    action = np.argmax(prediction_values)
    state, reward, done, info = take_skip_frame_step(env, action, 4, True)
    long_state = generate_stacked_state(long_state, state)

    step_counter+=1
    print("Steps: " +  str(step_counter) + " --- position: " + str(info['x_pos']))

env.close()


# state_list = []
# env.reset()
# action = 1
# for i in range(4):
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     state_list.append(state.copy())
#     env.render()
#
#
# pre_process_images = pre_process_images(state_list)