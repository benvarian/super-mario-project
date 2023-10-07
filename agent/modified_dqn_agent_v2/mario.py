from collections import deque
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt


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

    def create_network(self):
        '''Method to build the artificial neural network'''
        model = Sequential()
        model.add(
            Conv2D(64, (4, 4), strides=4, padding="same", input_shape=self.state_space)
        )
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(Activation("relu")) # Flatten into 1D vector for processing
        model.add(Flatten())

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())

        return model

    def update_target_network(self):
        '''Method to update target network with the modified main network'''
        self.target_network.set_weights(self.main_network.get_weights())

    def act(self, state, onGround):
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
        
    def plotFilters(self, conv_filter):
        fig, axes = plt.subplots(1, 3, figsize=(5,5))
        axes = axes.flatten()
        for img, ax in zip( conv_filter, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    def visualise_filter_v2(self):
        #Iterate thru all the layers of the model
        for layer in self.main_network.layers:
            if 'conv' in layer.name:
                filters, bias= layer.get_weights()
                print(layer.name, filters.shape)
                #normalize filter values between  0 and 1 for visualization
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)  
                print(filters.shape[3])
                axis_x=1
                #plotting all the filters
                for i in range(filters.shape[3]):
                #for i in range(6):
                    #get the filters
                    filt=filters[:,:,:, i]
                    self.plotFilters(filt)
                    
    def visualise_filters(self):
        #Visualizing the filters
        #plt.figure(figsize=(5,5))
        for layer in self.main_network.layers:
            if 'conv' in layer.name:
                weights, bias= layer.get_weights()
                print(layer.name, weights.shape)
                #normalize filter values between  0 and 1 for visualization
                f_min, f_max = weights.min(), weights.max()
                filters = (weights - f_min) / (f_max - f_min)  
                print(weights.shape[3])
                filter_cnt=1
                #plotting all the filters
                for i in range(filters.shape[3]):
                #for i in range(6):
                    #get the filters
                    filt=filters[:,:,:, i]
                    #plotting ecah channel
                    for j in range(filters.shape[0]):
                        #plt.figure( figsize=(5, 5) )
                        #f = plt.figure(figsize=(10,10))
                        ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.imshow(filt[:,:, 0]) # should be j
                        filter_cnt+=1
                plt.show()
    
    def visualise_feature(self):
        successive_feature_maps = self.main_network.predict(state)
        layer_names = [layer.name for layer in self.main_network.layers]
        
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            print(len(feature_map.shape))
            if True:
                
                #-------------------------------------------
                # Just do this for the conv / maxpool layers, not the fully-connected layers
                #-------------------------------------------
                n_features = feature_map.shape[-1]  # number of features in the feature map
                size       = feature_map.shape[ 0]  # feature map shape (1, size, size, n_features)
                
                # We will tile our images in this matrix
                display_grid = np.zeros((size, size * n_features))
                
                #-------------------------------------------------
                # Postprocess the feature to be visually palatable
                #-------------------------------------------------
                for i in range(n_features):
                    x  = feature_map[0, :, :, 0]
                    x -= x.mean()
                    x /= x.std ()
                    x *=  64
                    x += 128
                    x  = np.clip(x, 0, 255).astype('uint8')
                    display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

                #-----------------
                # Display the grid
                #-----------------

                scale = 20. / n_features
                plt.figure( figsize=(scale * n_features, scale) )
                plt.title ( layer_name )
                plt.grid  ( False )
                plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
    
    def visualize_feature_maps_for_state(self, state):

        # Iterate through all layers of the model
        for layer in self.main_network.layers:
            if 'conv' in layer.name:
                # Create a sub-model that extracts feature maps from the current layer
                feature_map_model = Sequential()
                feature_map_model.add(layer)

                # Predict feature maps for the preprocessed state
                feature_maps = feature_map_model.predict(state)

                # Get the number of feature maps
                num_feature_maps = feature_maps.shape[-1]

                # Create a subplot grid to display the feature maps
                rows = int(np.sqrt(num_feature_maps))
                cols = num_feature_maps // rows

                # Plot the feature maps
                plt.figure(figsize=(10, 10))
                for i in range(num_feature_maps):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
                    plt.axis('off')

                plt.title(f'Layer: {layer.name}')
                plt.show()
                        
                
    def visualize_feature_maps_for_state_v2(self, state):
        # Create a separate model for visualization with the same architecture as the main network
        visualize_model = Sequential()
        visualize_model.add(Conv2D(64, (4, 4), strides=4, padding="same", input_shape=self.state_space))
        visualize_model.add(Activation("relu"))
        visualize_model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        visualize_model.add(Activation("relu"))
        visualize_model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        visualize_model.add(Activation("relu"))

        # Predict feature maps for the preprocessed state
        feature_maps = visualize_model.predict(state)

        # Get the number of feature maps
        num_feature_maps = feature_maps.shape[-1]

        # Create a subplot grid to display the feature maps
        rows = int(np.sqrt(num_feature_maps))
        cols = num_feature_maps // rows

        # Plot the feature maps
        plt.figure(figsize=(10, 10))
        for i in range(num_feature_maps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        plt.title('Feature Maps Visualization')
        plt.show()
    
    def visualize_feature_maps_for_all_layers(self, state):
        # Create a separate model for visualization with the same architecture as the main network
        visualize_model = Sequential()

        # Iterate through all layers of the main network
        for layer in self.main_network.layers:
            visualize_model.add(layer)
            if 'conv' in layer.name:
                # Predict feature maps for the preprocessed state
                feature_maps = visualize_model.predict(state)

                # Get the number of feature maps
                num_feature_maps = feature_maps.shape[-1]

                # Create a subplot grid to display the feature maps
                rows = int(np.sqrt(num_feature_maps))
                cols = num_feature_maps // rows

                # Plot the feature maps
                plt.figure(figsize=(10, 10))
                for i in range(num_feature_maps):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
                    plt.axis('off')

                plt.title(f'Layer: {layer.name}')
                plt.show()
        
                
from collections import deque

import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)
num_episodes = 3000000
num_timesteps = 400000

batch_size = 64
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
        #dqn.visualise_filter_v2()
        #dqn.visualise_feature()
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
        
        #dqn.visualize_feature_maps_for_all_layers(state)

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
    dqn.save_agent(f"artifacts/model_{i}.h5")

env.close()
