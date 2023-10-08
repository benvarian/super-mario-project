import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state

from mario_visualise import MarioAgent

from collections import deque
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque

import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)


dqn = MarioAgent(state_space, action_space)

dqn.load_agent("artifacts/model_0_vis.h5")


total_reward = 0
while True:
    print("HELLO")
    done = False
    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)
    onGround = 79
    
    i = 0

    while not done:
        if i > 110:
            dqn.visualize_filters()
            dqn.visualize_feature_maps()
        i +=1
        print("HELLO")
        env.render()
        action = dqn.act(state, onGround)
        next_state, reward, done, info = env.step(action)

        onGround = info["y_pos"]

        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1, 80, 88, 1)
        state = next_state

env.close()