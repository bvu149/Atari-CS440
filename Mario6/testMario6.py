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
from helper_file import *


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

warnings.simplefilter("ignore", lineno=148)


current_user = getpass.getuser()
if (current_user == "gryslik"):
    model_path = '/Users/gryslik/gitRepos/qlearning/test_code/mario/models6-DDQN/'
else:
    model_path = '/home/ubuntu/data/code/mario/models6-DDQN/'


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #2-2, 1-1
env = JoypadSpace(env, RIGHT_ONLY)

all_files = os.listdir(model_path)

if "travel_distance.csv" in all_files:
    models_processed = pd.read_csv(model_path+"travel_distance.csv")['model_name'].values
    models_to_compute = [x for x in all_files if (x not in models_processed and x not in ".DS_Store" and  x not in "travel_distance.csv")]
else:
    models_to_compute = [item for item in all_files if (item not in "travel_distance.csv" and item not in ".DS_Store")]


num_frames_to_stack = 4
num_frames_to_collapse = 4
for idx, model_name in enumerate(models_to_compute):
    print("Processing index/name: " + str(idx) + " --- " + model_name)
    try:
        time_start = time.time()
        model = tf.keras.models.load_model(model_path + model_name)

        long_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
        done = False;
        step_counter = 0
        while not done and step_counter < 500: # Now we need to take the same action every 4 steps
            prediction_values = model.predict(np.expand_dims(long_state, axis=0).astype('float16'))
            action = np.argmax(prediction_values)
            state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_collapse, False)
            long_state = generate_stacked_state(long_state, state)
            step_counter+=1

        all_files = os.listdir(model_path)
        if "travel_distance.csv" not in all_files:
            print("In here")
            with open(model_path + "travel_distance.csv", 'a') as f:
                pd.DataFrame([[model_name, info['x_pos']]], columns=['model_name', 'travel_distance']).to_csv(model_path + "travel_distance.csv", header=True, mode='w')
        else:
            print("in here 2")
            with open(model_path + "travel_distance.csv", 'a') as f:
                pd.DataFrame([[model_name, info['x_pos']]], columns=['model_name', 'travel_distance']).to_csv(model_path+"travel_distance.csv", header=False, mode='a')

        time_end = time.time()
        print("This took: " + str(int(time_end - time_start)) + " seconds. Ended at position: " + str(info['x_pos']))
        print('-----------------------------------------------------------------------------------------------------------')
    except:
        print("Episode: " + str(idx) + " failed!")
env.close()