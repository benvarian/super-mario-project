import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# Clear Joypad space to silent one of the warnings
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
# changes to env to make it work better
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")


# TrainAndLogg is a callback that saves the model every x number steps to the current directory
class TrainAndLogg(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLogg, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(
                os.path.join(self.save_path, f"rl_model_PPO25_{self.n_calls}.zip")
            )
        return True


CHECK_DIR = "./train"
LOG_DIR = "./logs"

state = env.reset()
state, reward, done, info = env.step([5])


model_old = PPO.load("./train/rl_model_10000000.zip")
# initiliase the callback
callback = TrainAndLogg(check_freq=100000, save_path=CHECK_DIR)
# change model_old to simply model and you'll have a new model to train
# model = PPO(
#     "CnnPolicy",
#     env,
#     verbose=1,
#     tensorboard_log=LOG_DIR,
#     learning_rate=0.00004,
#     n_steps=1024,
# )
# model_old.learn(total_timesteps=10000000, callback=callback)
# load up the trained model and set it to the current env
model_old.set_env(env)
# train the model more than it already has
model_old.learn(total_timesteps=6000000, callback=callback)