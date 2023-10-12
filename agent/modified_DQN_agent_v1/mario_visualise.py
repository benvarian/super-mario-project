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



class MarioAgent:
    def __init__(self, state_space, action_space):
        # Store memory of agent
        self.state_space = state_space # Stores the size of the state space; the input dimensions
        self.action_space = action_space # Stores the number of possible actions
        self.state_memory = deque(maxlen=7000) # Used for experience replay; this (1) decouples sequential experiences (2) ensures the model doesn't 'forget' prior learned experiences
        self.action = 1 # Action agent chooses
        self.gamma = 0.9 # The discount factor of rewards; a high gamma prioritises longer term rewards

        # Control exploration vs exploitation behaviour
        self.epsilon = 1 # Exploration rate: probability of taking random action; to change over time
        self.max_epsilon = 1 # Exploration rate at beginning
        self.min_epsilon = 0.03 # Agent should always have some exploration
        self.decay_epsilon = 0.0003 # The rate epsilon decays over time; exploration should occur less as time goes on

        # Building Neural Networks for Agent
        self.main_network = self.create_network() # Responsible for learning and updating Q-values
        self.target_network = self.create_network() # Responsible for maintainiing stable Q-values
        self.update_target_network()
        
        self.feature_map_model = Sequential(
            layers=self.main_network.layers[:4]  # Assuming the first 4 layers are convolutional
        )
        self.feature_maps = []  # Store feature maps during prediction

    def create_network(self):
        '''Method to build the artificial neural network'''
        model = Sequential()
        model.add(Conv2D(64, (4, 4), strides=4, padding="same", input_shape=self.state_space))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(Activation("relu"))
        
        model.add(Conv2D(32, (2, 2), strides=1, padding="same"))
        model.add(Activation("relu"))
        
        model.add(Flatten()) # Flatten into 1D vector for processing

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())

        return model

    def update_target_network(self):
        '''Method to update target network with the modified main network'''
        self.target_network.set_weights(self.main_network.get_weights())

    def act(self, state, onGround):
        
        self.feature_maps = self.feature_map_model.predict(state)
        if onGround < 83:
            print("On Ground")
            if random.uniform(0, 1) < self.epsilon:
                self.action = np.random.randint(self.action_space)
                return self.action
            Q_value = self.main_network.predict(state)
            self.action = np.argmax(Q_value[0])
        else:
            print("Not on Ground")

        return self.action

    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_epsilon * episode)

    # train the network
    def train(self, batch_size):
        # minibatch from memory
        minibatch = random.sample(self.state_memory, batch_size)

        # Get variables from batch so we can find q-value
        for state, action, reward, next_state, done in minibatch:
            target = self.main_network.predict(state)
            print("This is target", target)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(
                    self.target_network.predict(next_state)
                )

            self.main_network.fit(state, target, epochs=1, verbose=0)

    def add_state_to_memory(self, state, action, reward, next_state, done):
        self.state_memory.append((state, action, reward, next_state, done))

    def get_best_predicted_action(self, state):
        Q_values = self.main_network.predict(state)
        print("This is q values", Q_values)
        return np.argmax(Q_values[0])

    def load_agent(self, name):
        self.main_network = load_model(name)
        self.target_network = load_model(name)

    def save_agent(self, name):
        save_model(self.main_network, name)
        
    def visualize_filters(self):
    # Assuming your first layer is a Conv2D layer
        filters = self.main_network.layers[0].get_weights()[0]
        num_filters = filters.shape[3]

        plt.figure(figsize=(10, 10))
        for i in range(num_filters):
            plt.subplot(8, 8, i + 1)
            plt.imshow(filters[:, :, 0, i], cmap='viridis')
            plt.axis('off')
        plt.show()
        
    def visualize_feature_maps(self):
    # Convert self.feature_maps to a NumPy array
        feature_maps_array = np.array(self.feature_maps)

        num_feature_maps = feature_maps_array.shape[0]

        plt.figure(figsize=(10, 10))
        for i in range(num_feature_maps):
            plt.subplot(8, 8, i + 1)
            plt.imshow(feature_maps_array[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.show()

'''        
        
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)
#num_episodes = 1000000
#num_timesteps = 400000
num_episodes = 3
num_timesteps = 5000
batch_size = 32
DEBUG_LENGTH = 300

dqn = MarioAgent(state_space, action_space)
X_WORLD_POS = deque(maxlen=DEBUG_LENGTH)


for i in range(num_episodes):
    Return = 0
    done = False
    time_step = 0
    onGround = 79

    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)

    for t in range(num_timesteps):
        
        env.render()
        time_step += 1

        if t > 1 and X_WORLD_POS.count(X_WORLD_POS[-1]) > DEBUG_LENGTH - 50:
            action = dqn.act(state, onGround=79)
        else:
            action = dqn.act(state, onGround)

        print("ACTION IS" + str(action))

        next_state, reward, done, info = env.step(action)
        #done = terminated or truncated
        onGround = info["y_pos"]
        X_WORLD_POS.append(info["x_pos"])

        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1, 80, 88, 1)

        dqn.add_state_to_memory(state, action, reward, next_state, done)
        state = next_state

        Return += reward
        print(
            "Episode is: {}\nTotal Time Step: {}\nCurrent Reward: {}\nEpsilon is: {}".format(
                str(i), str(time_step), str(Return), str(dqn.epsilon)
            )
        )
        if done:
            break

        if len(dqn.state_memory) > batch_size and i > 0:
            dqn.train(batch_size)

    dqn.update_epsilon(i)
    dqn.update_target_network()
    dqn.save_agent(f"artifacts/model_{i}_vis.h5")

env.close()

'''