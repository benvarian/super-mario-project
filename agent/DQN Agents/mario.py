'''Training architecture for DQN agent'''

# Import libraries
from collections import deque
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from utils import preprocess_state


class MarioAgent:
    def __init__(self, state_space, action_space):
        # Store memory of agent
        self.state_space = (
            state_space  # Stores the size of the state space; the input dimensions
        )
        self.action_space = action_space  # Stores the number of possible actions
        self.state_memory = deque(
            maxlen=7000
        )  # Used for experience replay; this (1) decouples sequential experiences (2) ensures the model doesn't 'forget' prior learned experiences
        self.action = 1  # Action agent chooses
        self.gamma = 0.9  # The discount factor of rewards; a high gamma prioritises longer term rewards

        # Control exploration vs exploitation behaviour
        self.epsilon = 1  # Exploration rate: probability of taking random action; to change over time
        self.max_epsilon = 1  # Exploration rate at beginning
        self.min_epsilon = 0.01  # Agent should always have some exploration
        self.decay_epsilon = 0.0001  # The rate epsilon decays over time; exploration should occur less as time goes on

        # Building Neural Networks for Agent
        self.main_network = (
            self.create_network()
        )  # Responsible for learning and updating Q-values
        self.target_network = (
            self.create_network()
        )  # Responsible for maintainiing stable Q-values
        self.update_target_network()
        
        self.ONGROUND = 83 # on ground constant

    def create_network(self):
        """Method to build the artificial neural network"""
        model = Sequential()
        model.add(
            Conv2D(64, (4, 4), strides=4, padding="same", input_shape=self.state_space)
        )
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(Activation("relu"))  # Flatten into 1D vector for processing
        model.add(Flatten())

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())

        return model

    def update_target_network(self):
        """Method to update target network with the modified main network"""
        self.target_network.set_weights(self.main_network.get_weights())

    def act(self, state, onGround):
        '''Method to return an action'''
        if onGround < self.ONGROUND: # if on ground
            if random.uniform(0, 1) < self.epsilon: # To explore new action
                self.action = np.random.randint(self.action_space)
                return self.action
            # Else predict action
            Q_value = self.main_network.predict(state)
            self.action = np.argmax(Q_value[0])

        return self.action # Return action

    def update_epsilon(self, episode):
        '''Method to decay epsilon (exploration threshold)'''
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_epsilon * episode)

    def train(self, batch_size):
        '''Method to train the model using samples stored (experience replay)'''
        # Sample batch for experience replay
        minibatch = random.sample(self.state_memory, batch_size)

        # Get target Q-value
        for state, action, reward, next_state, done in minibatch:
            target = self.main_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(
                    self.target_network.predict(next_state)
                )

            self.main_network.fit(state, target, epochs=1, verbose=0)

    def add_state_to_memory(self, state, action, reward, next_state, done):
        '''Method to add state for memory for experience replay'''
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

    def plotFilters(self, conv_filter):
        fig, axes = plt.subplots(1, 3, figsize=(5, 5))
        axes = axes.flatten()
        for img, ax in zip(conv_filter, axes):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_filters(self):
        '''Visualise kernals used for convolution'''
        for layer in self.main_network.layers:
            if "conv" in layer.name:
                filters, bias = layer.get_weights()
                print(layer.name, filters.shape)
                # normalize filter values between  0 and 1 for visualization
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)
                print(filters.shape[3])
                axis_x = 1
                # plotting all the filters
                for i in range(filters.shape[3]):
                    filt = filters[:, :, :, i]
                    self.plotFilters(filt)

    def plot_conv_layer1(self):
        successive_feature_maps = self.main_network.predict(state)
        layer_names = [layer.name for layer in self.main_network.layers]

        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            print(len(feature_map.shape))
            if True:
                n_features = feature_map.shape[-1]
                size = feature_map.shape[0] 

                display_grid = np.zeros((size, size * n_features))

                for i in range(n_features):
                    x = feature_map[0, :, :, 0]
                    x -= x.mean()
                    x /= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype("uint8")
                    display_grid[:, i * size : (i + 1) * size] = x
                scale = 20.0 / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect="auto", cmap="viridis")

    def plot_conv_layer2(self, state):
        visualize_model = Sequential()

        for layer in self.main_network.layers:
            visualize_model.add(layer)
            if "conv" in layer.name:
                feature_maps = visualize_model.predict(state)
                num_feature_maps = feature_maps.shape[-1]
                rows = int(np.sqrt(num_feature_maps))
                cols = num_feature_maps // rows
                plt.figure(figsize=(10, 10))
                for i in range(num_feature_maps):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
                    plt.axis("off")

                plt.title(f"Layer: {layer.name}")
                plt.show()
