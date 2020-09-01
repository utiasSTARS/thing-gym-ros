import sys
import time

import gym

from thing_gym_ros.envs.thing_ros_generic import Talker, ThingRosEnv


num_steps = 1000

env = ThingRosEnv(img_in_state=True, depth_in_state=True, dense_reward=False, grip_in_action=True,
                  default_grip_state='o', num_objs=1, moving_base=True)
env.seed(0)
env.render()

import ipdb; ipdb.set_trace()

obs = env.reset()

for i in range(num_steps):
    act = env.action_space.sample()
    next_obs, rew, done, info = env.step(act)

    # env.render()
    obs = next_obs