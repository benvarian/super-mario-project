"""Run DQN mario agent"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state
from mario import MarioAgent

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)
dqn = MarioAgent(state_space, action_space)

dqn.load_agent("train/model_1.h5")

LAST_WORLD_X_Y = [(0, 0)] * 30

while True:
    done = False
    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)
    onGround = 80

    while not done:
        env.render()
        action = dqn.act(state, onGround)
        next_state, reward, done, info = env.step(action)
        LAST_WORLD_X_Y.pop(0)
        LAST_WORLD_X_Y.append((info["x_pos"], info["y_pos"]))

        # Failsafe: if Mario stays in same position for too long
        if LAST_WORLD_X_Y.count(LAST_WORLD_X_Y[0]) == len(LAST_WORLD_X_Y):
            obs, reward, done, info = env.step(1)
            obs, reward, done, info = env.step(4)

        onGround = info["y_pos"]

        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1, 80, 88, 1)
        state = next_state
