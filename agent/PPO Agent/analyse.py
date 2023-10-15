import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")
# load up the PPO model from a file 
# please refer to README.md if you are having problems with this
model = PPO.load("./train/rl_model.zip")

# loop through all of the available actions and render it to the env
obs = env.reset()
done = False
while not done:
    action, state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
