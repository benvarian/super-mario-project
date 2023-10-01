# Install pytorch
# poetry add torch torchvision torchaudio
# install stable baselines for RL algorithm
# poetry add stable-baselines3[extra]
# pip3 install stable-baselines --use-deprecated=backtrack-on-build-failures
# poetry add  stable-baselines3[extra]

# https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values

# new
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import frame_stack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

# rl model
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")
# check_env(env)


# done = True
class TrainAndLogg(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLogg, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.data = []

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        state, reward, done, info = env.step([5])
        self.data.append((done, info))
        if self.n_calls % self.check_freq == 0:
            self.model.save(
                os.path.join(self.save_path, f"rl_model_{self.n_calls}.zip")
            )
            with open(f"info_model_{self.n_calls}.txt", "w") as file:
                file.write(str(self.data))
        return True

CHECK_DIR = "./train"
LOG_DIR = "./logs"

# for step in range(100000):
#     if done:
#         env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
# env.close()

state = env.reset()
# print(state.shape)
state, reward, done, info = env.step([5])
# plt.imshow(state[0])
# plt.figure(figsize=(20, 16))
# for idx in range(state.shape[3]):
#     plt.subplot(1, 4, idx + 1)
#     plt.imshow(state[0][:, :, idx])
# plt.show()


callback = TrainAndLogg(check_freq=512, save_path=CHECK_DIR)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.0003,
    n_steps=512,
)
model.learn(total_timesteps=100000, callback=callback)