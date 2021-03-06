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
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from helper_file import *

warnings.simplefilter("ignore", lineno=148)

current_user = getpass.getuser()
if (current_user == "gryslik"):
    model_path = '/Users/gryslik/gitRepos/qlearning/test_code/mario/models6-DDQN/'
else:
    model_path = '/home/ubuntu/data/code/mario/models6-DDQN/'


class DQN:
    def __init__(self, env, single_frame_dim,  num_frames_to_stack, old_model_filepath=None):
        self.env = env
        self.memory = collections.deque(maxlen=100000) #5000

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999997
        self.learning_rate = 0.00003
        self.burnin = 20000 #2000
        self.update_target_step_count = 4000 #3000
        self.num_steps_since_last_update = 0
        self.single_frame_dim = single_frame_dim
        self.num_frames_to_stack = num_frames_to_stack

        if(old_model_filepath==None):
            self.model = self.create_model()  # Will do the actual predictions
            self.target_model = self.create_model()  # Will compute what action we DESIRE from our model
        else:
            self.model = tf.keras.models.load_model(old_model_filepath)
            self.target_model = tf.keras.models.load_model(old_model_filepath)
        # Otherwise we are changing the goal at each timestep.

    def update_target_model(self, current_episode, current_step, update_by_episode_count):

        self.num_steps_since_last_update +=1 #just did a step, update the counter
        #update if new episode and parameter set or by episode count
        if (update_by_episode_count and current_step == 0) or (not update_by_episode_count and (self.num_steps_since_last_update == self.update_target_step_count)): # for smaller problems you might want to update just by episode. For larger problems, usually cap it at 5k or max for episode or whatever heuristic you prefer.
            print("Updating_target_model at episode/step: " + str(current_episode) + " / " +str(current_step))
            self.target_model.set_weights(self.model.get_weights())
            self.num_steps_since_last_update = 0

    def create_model(self):
        model = tf.keras.Sequential()
        #model.add(tf.keras.layers.Conv2D(32, 8, 8, input_shape=self.env.observation_space.shape, activation="relu"))
        model.add(tf.keras.layers.Conv2D(32, 8, 8, input_shape=(self.single_frame_dim[0], self.single_frame_dim[1], self.num_frames_to_stack), activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 4, 4, activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=64):
        if len(self.memory) < self.burnin:
            return
        samples = random.sample(self.memory, batch_size)
        all_states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, self.single_frame_dim[0],  self.single_frame_dim[1], num_frames_to_stack))
        all_actions = np.reshape([x[1] for x in samples], (batch_size,))
        all_rewards = np.reshape([x[2] for x in samples], (batch_size,))
        all_new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, self.single_frame_dim[0], self.single_frame_dim[1], num_frames_to_stack))
        all_dones = np.reshape([x[4] for x in samples], (batch_size,))

        all_targets = np.array(self.model.predict_on_batch(all_states.astype('float16')))  # this is what we will update
        Q_0 = np.array(self.model.predict_on_batch(all_new_states.astype('float16')))  # This is what we use to find what max action we should take
        Q_target = np.array(self.target_model.predict_on_batch(all_new_states.astype('float16')))  # This is the values we will combine with max action to update the target
        max_actions = np.argmax(Q_0, axis=1)  # This is the index we will use to take from Q_target
        max_Q_target_values = Q_target[np.arange(len(Q_target)), np.array(max_actions)]  # The target will be updated with this.
        all_targets[np.arange(len(all_targets)), np.array(all_actions)] = all_rewards + self.gamma * max_Q_target_values * (~all_dones)  # Actually due the update

        self.model.train_on_batch(all_states.astype('float16'), all_targets)  # reweight network to get new targets

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            reshaped_state = np.expand_dims(state, axis=0).astype('float16')
            return np.argmax(self.model.predict(reshaped_state)[0]) #The predict returns a (1,7)

    def save_model(self, fn):
        self.model.save(fn)


#######################################################################################
# Initialize environment and parameters
#######################################################################################
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
raw_image_dim = pre_process_image(env.reset()).shape
num_episodes = 8000
num_frames_to_collapse = 4
num_frames_to_stack = 4
my_agent = DQN(env=env,single_frame_dim=raw_image_dim,num_frames_to_stack=num_frames_to_stack)
totalreward = []
steps = []
flag_result = []
final_x_position = []

for episode in range(num_episodes):
    print("----------------------------------------------------------------------------")
    print("Episode: " + str(episode) + " started")
    time_start = time.time()
    cur_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
    episode_reward = 0
    step = 0
    done = False
    current_x_position = []

    while not done:
        if(step % 100 == 0):
            print("At step: " + str(step))

        action = my_agent.act(cur_state)
        my_agent.update_target_model(episode, step, False)  # self.update_target_step_count steps or every episode. Here every episode

        new_state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_collapse) #take a step for 4 actions
        new_state = generate_stacked_state(cur_state, new_state) #make the new state (120x128x4) to account for frame stacking
        step += 1

        #Compute the new reward if stuck
        if len(current_x_position) > 250:
            avg_position = round(np.array(current_x_position[-250:]).mean(), 0)
            current_position = info['x_pos']
            if current_position - avg_position < 1:  ## No real movement
                reward += -5 #make it negative so that it doesn't get stuck and tries new things
                reward = max(reward, -15) #can't go below -15

        #Add to memory
        my_agent.remember(cur_state, action, reward, new_state, done)

        #fit the model
        my_agent.replay()

        #set the current_state to the new state
        cur_state = new_state

        episode_reward += reward
        current_x_position.append(info['x_pos'])

        if info['flag_get']:
            print("Breaking due to getting flag!")
            break
        if step > 3000:
            print("Breaking due to out of steps.")
            break

    totalreward.append(episode_reward)
    steps.append(step)
    flag_result.append(info['flag_get'])
    final_x_position.append(current_x_position[-1])
    if info['flag_get']:
        print("Episode: " + str(episode) + " -- SUCCESS -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1]))
        my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))
    else:
        print("Episode: " + str(episode) + " -- FAILURE -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1]))
        if episode % 10 == 0:
            my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))

    time_end = time.time()
    tf.keras.backend.clear_session()
    print("Episode: " + str(int(episode)) + " completed in steps/time/avg_running_reward: " + str(steps[-1]) + " / " + str(int(time_end - time_start)) + " / " + str(np.array(totalreward)[-100:].mean()))
    print("----------------------------------------------------------------------------")
env.close()

results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
results_df['steps_taken'] = steps
results_df['flag_get'] = flag_result
results_df['x_pos'] = final_x_position
results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()

results_df.to_csv(model_path + "training_results.csv")


