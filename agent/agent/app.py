# Install pytorch   
#poetry add torch torchvision torchaudio
# install stable baselines for RL algorithm
#poetry add stable-baselines3[extra]
# pip3 install stable-baselines --use-deprecated=backtrack-on-build-failures
# poetry add  stable-baselines3[extra]

# https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym


# PREPROCESS ENVIRONMENT
# Import gray scaling wrapper (less information for agent)
from gym.wrappers import GrayScaleObservation
#Import vectorisation wrappers (VecFrameStack deals with multiple frame capture - allows agent to see direction enemies move)
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# Import OS for file path management
import OS
# Import PPO for algos
from stable_baselines3 import PPO
# Import base callback for saving models
from stable_baselines3.common.callbacks import BaseCall


# 1. Create base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
#env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Greyscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside dummy environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')


'''
done = True
env.reset()

for step in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        state = env.reset()
        
env.close()

'''    
done = True
for step in range(100000):
    if done:
        env.reset()
    # Do random actions
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # Show game on screen
    env.render()
# Close the game
env.close()

# AI model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

#train ai model
model.learn(total_timesteps=1000000)