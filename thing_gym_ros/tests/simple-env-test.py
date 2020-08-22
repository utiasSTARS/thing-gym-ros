import sys

import gym

from thing_gym_ros.envs.thing_ros_generic import Talker, ThingRosEnv


num_steps = 1000

env = ThingRosEnv()
env.seed(0)
obs = env.reset()

for i in range(num_steps):
    act = env.action_space.sample()
    next_obs, rew, done, info = env.step(act)

    # env.render()
    obs = next_obs