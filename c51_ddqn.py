#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure, io
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time
import math
import os
import pandas as pd
import cv2
import csv
from PIL import Image

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks

#DEATHMATCH_ACTION5_NAME = [
#    "ATTACK",
#    "MOVE_FORWARD",
#    "MOVE_BACKWARD",
#    "TURN_LEFT",
#    "TURN_RIGHT"
#]

DEATHMATCH_ACTION5_NAME = [
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT"
]

def preprocessImg(img, size):
    
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size, mode='constant')
    img = skimage.color.rgb2gray(img)

    return img

def ResizeImg(img, size):
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size, mode='constant')
    return img

bTrain = True
bUseImitation = False
bRecordSamples = False
nMaxSamples = 1000
nSamples = 0
gameCfg = "./scenarios/deathmatch_7action.cfg"

# This is for saving model of imitation learning.
model_path = "../ViZDoom-models/CarCloneModel-deathmatch-50000-epoch10-5action-256x256-modify1/"

class CNNAction:
    def __init__(self, gameName):
        model_json = model_path + "test_model.json"
        model_h5 = model_path + "test_model.h5"
        with open(model_json, 'r') as jfile:
            self.model = model_from_json(json.load(jfile))

        self.model.compile("adam", "categorical_crossentropy")
        self.model.load_weights(model_h5)
        self.imgList = []
        self.model.summary()

        self.w1 = 256
        self.h1 = 256
        self.inputW = 128
        self.inputH = 128

        self.frame_per_action = 4
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.observe = 2000

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts
        
        # sample picture number
        dataPath = "ImitationData/" + gameName
        if not os.path.exists(dataPath):
            os.mkdir(dataPath)
        imgPath = dataPath + "/img"
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        self.sampleNum = 0
        self.imgPath = imgPath
        self.dataPath = dataPath
        self.cvsPath = dataPath + "/test.csv"
        self.sampleCSVFile = open(self.cvsPath, "w")
        self.sampleCSVWriter = csv.writer(self.sampleCSVFile)
        self.sampleCSVWriter.writerow(["name", "action", "action_name"])
        
    def GenerateSamples(self, screen, action):
        self.sampleNum = self.sampleNum + 1
        t = time.time()
        now = int(round(t*1000))
        timeStr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(now/1000))
        savedFileName = "%s/doom-%s-%d.jpg" % (self.imgPath, timeStr, self.sampleNum)
        self.sampleCSVWriter.writerow([savedFileName, action, DEATHMATCH_ACTION5_NAME[action]])
        self.sampleCSVFile.flush()

        # skimage.io.imsave("hy.jpg", screen.transpose(1, 2, 0))
        dst = ResizeImg(screen, (256, 256))
        skimage.io.imsave(savedFileName, dst)
        return

    def next_action(self, state, save_graph=False):
        action_id = self.f_eval(state)
        return action_id

    def reset(self):
        pass
        # prev_state is only used for evaluation, so has a batch size of 1
        # self.prev_state = self.init_state_e

    def prepare_f_eval_args(self, state):
        """
        Prepare inputs for evaluation.
        """
        screen = np.float32(state)
        return screen

    def f_eval(self, state):
        screen = self.prepare_f_eval_args(state)
        img = screen
        # print (img.shape)

        img = cv2.resize(img.transpose(1, 2, 0), (self.w1, self.h1), interpolation=cv2.INTER_AREA)
        self.imgList.append(img)
#        if len(self.imgList) < 4:
#            return 0

#        img1Int = self.imgList[0].transpose(2, 1, 0).astype(int)

        img1 = array_to_img(self.imgList[0].astype(int))
#        img2 = array_to_img(self.imgList[1].astype(int))
#        img3 = array_to_img(self.imgList[2].astype(int))
#        img4 = array_to_img(self.imgList[3].astype(int))

        w = self.w1
        h = self.h1
        merge_img = Image.new('RGB', (w, h), 0xffffff)
        merge_img.paste(img1, (0, 0))
#        merge_img.paste(img2, (w, 0))
#        merge_img.paste(img3, (0, h))
#        merge_img.paste(img4, (w, h))
        merge_img.save("hy.jpg")

        merge_img = merge_img.resize((self.inputW, self.inputH))

        img5 = img_to_array(merge_img).transpose(0, 1, 2)
        img5 = img5.astype("float32")
        img5 = (img5 * (1. / 255)) - 0.5
        imgs = img5[None, :, :, :]
        # print (imgs.shape)

        action_id = self.model.predict(imgs, batch_size=1)
        action_list = np.argsort(-action_id, axis=1)

        self.imgList.pop(0)
        return int(action_list[0][0])

class C51Agent:

    def __init__(self, state_size, action_size, num_atoms, gameName):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 2000
        self.explore = 100000 # orig: 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 100  # Number of timesteps between training interval

        # Initialize Atoms
        self.num_atoms = num_atoms  # 51 for C51
        self.v_max = 30  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -10  # -0.1*26 - 1 = -3.6
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()
        self.max_memory = 100000 # orig: 50000  # number of previous transitions to remember

        # Models for value distribution
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts

        # sample picture number
        dataPath = "ImitationData/" + gameName
        if not os.path.exists(dataPath):
            os.mkdir(dataPath)
        imgPath = dataPath + "/img"
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        self.sampleNum = 0
        self.imgPath = imgPath
        self.dataPath = dataPath
        self.cvsPath = dataPath + "/test.csv"
        self.sampleCSVFile = open(self.cvsPath, "w")
        self.sampleCSVWriter = csv.writer(self.sampleCSVFile)
        self.sampleCSVWriter.writerow(["name", "action", "action_name"])

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def GenerateSamples(self, screen, action):
        self.sampleNum = self.sampleNum + 1
        t = time.time()
        now = int(round(t*1000))
        timeStr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(now/1000))
        savedFileName = "%s/doom-%s-%d.jpg" % (self.imgPath, timeStr, self.sampleNum)
        self.sampleCSVWriter.writerow([savedFileName, action, DEATHMATCH_ACTION5_NAME[action]])
        self.sampleCSVFile.flush()

        # skimage.io.imsave("hy.jpg", screen.transpose(1, 2, 0))
        dst = ResizeImg(screen, (256, 256))
        skimage.io.imsave(savedFileName, dst)
        return

    def get_action(self, state, bTrain=True):
        """
        Get action from model using epsilon-greedy policy
        """
        if bTrain:
            if np.random.rand() <= self.epsilon:
                action_idx = random.randrange(self.action_size)
            else:
                action_idx = self.get_optimal_action(state)
        else:
            action_idx = self.get_optimal_action(state)

        return action_idx

    def get_optimal_action(self, state):
        """Get optimal action for a state
        """
        z = self.model.predict(state)  # Return a list [1x51, 1x51, 1x51]

        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)

        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):

        # Check any kill count orig reward:
        # if (misc[0] > prev_misc[0]):
        #     r_t = r_t + 1

        # if (misc[1] < prev_misc[1]):  # Use ammo
        #     r_t = r_t - 0.1

        # if (misc[2] < prev_misc[2]):  # Loss HEALTH
        #     r_t = r_t - 0.1

        # hy modify
        if (misc[0] > prev_misc[0]):  # kill
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]):  # Use ammo
            r_t = r_t - 0.2

        if (misc[2] < prev_misc[2]):  # Loss HEALTH
            r_t = r_t - 0.1

        return r_t

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        state_inputs = np.zeros(((num_samples,) + self.state_size)) 
        next_states = np.zeros(((num_samples,) + self.state_size)) 
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            state_inputs[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        z = self.model.predict(next_states) # Return a list [32x51, 32x51, 32x51]
        z_ = self.model.predict(next_states) # Return a list [32x51, 32x51, 32x51]

        # Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
        q = q.reshape((num_samples, action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]: # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, epochs=1, verbose=0)

        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    gameCfgFile = os.path.basename(gameCfg)
    gameName, extension = os.path.splitext(gameCfgFile)

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config(gameCfg)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4  # We stack 4 frames

    # C51
    num_atoms = 51

    state_size = (img_rows, img_cols, img_channels)

    if bUseImitation:
        agent = CNNAction(gameName)
    else:
        agent = C51Agent(state_size, action_size, num_atoms, gameName)

        agent.model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)
        agent.target_model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)

        if not bTrain:
            file = "./models/" + "c51_ddqn_" + gameName + ".h5"
            agent.load_model(file)

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    x_t = game_state.screen_buffer  # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2)  # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4

    is_terminated = game.is_episode_finished()

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not game.is_episode_finished():

        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        if bUseImitation:
            action_idx = agent.next_action(game_state.screen_buffer)
        else:
            action_idx = agent.get_action(s_t, bTrain)
        
        if not bTrain and bRecordSamples:
            agent.GenerateSamples(game_state.screen_buffer, action_idx)
            nSamples += 1
            if nSamples > nMaxSamples:
                break

        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()  # each frame we get reward of 0.1, so 4 frames will be 0.4

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            # print ("Episode Finish ", misc)
            print ("Episode: lifetime(%d) ammo(%d) kills(%d)" % (life, misc[1], misc[0]))
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        if bUseImitation:
            r_t = 0
        else:
            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # update the cache
        prev_misc = misc

        if not bUseImitation:
            if bTrain:
                # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
                agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

                # Do the training
                if t > agent.observe and t % agent.timestep_per_train == 0:
                    loss = agent.train_replay()
        else:
            sleep(0.01)

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if not bUseImitation:
            if t % 10000 == 0 and bTrain:
                file = "./models/" + "c51_ddqn_" + gameName + ".h5"
                print("Now we save model: %s" %(file))
                agent.model.save_weights(file, overwrite=True)

            # print info
            state = ""
            if t <= agent.observe:
                state = "observe"
            elif t > agent.observe and t <= agent.observe + agent.explore:
                state = "explore"
            else:
                state = "train"
        else:
            state = "observe"

        if (is_terminated):
            if bUseImitation:
                print("TIME", t, "/ GAME", GAME, "/ ACTION", action_idx,
                      "/ LIFE", max_life, "/ LOSS", loss)
            else:
                print("TIME", t, "/ GAME", GAME, "/ STATE", state,
                      "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, 
                      "/ REWARD", r_t, "/ LIFE", max_life, "/ LOSS", loss)
            
            # Training times.
            if GAME > 5000:
                break

            # Save Agent's Performance Statistics
            if bUseImitation:
                if GAME % agent.stats_window_size == 0:
                    print("Update Rolling Statistics")
                    agent.mavg_score.append(np.mean(np.array(life_buffer)))
                    agent.var_score.append(np.var(np.array(life_buffer)))
                    agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                    agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                    # Reset rolling stats buffer
                    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

                    # Write Rolling Statistics to file
                    with open("statistics/imitation_stats.txt", "w") as stats_file:
                        stats_file.write('Game: ' + str(GAME) + '\n')
                        stats_file.write('Max Score: ' + str(max_life) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                        stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')
            else:
                if GAME % agent.stats_window_size == 0 and t > agent.observe:
                    print("Update Rolling Statistics")
                    agent.mavg_score.append(np.mean(np.array(life_buffer)))
                    agent.var_score.append(np.var(np.array(life_buffer)))
                    agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                    agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                    # Reset rolling stats buffer
                    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

                    # Write Rolling Statistics to file
                    file = "./statistics/" + "c51_ddqn_stats_" + gameName + ".txt"
                    with open(file, "w") as stats_file:
                        stats_file.write('Game: ' + str(GAME) + '\n')
                        stats_file.write('Max Score: ' + str(max_life) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                        stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

