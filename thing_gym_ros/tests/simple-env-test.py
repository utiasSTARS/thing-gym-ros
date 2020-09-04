import sys
import time

import gym

from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


num_steps = 1000

env = ThingRosMBEnv(img_in_state=True, depth_in_state=True, dense_reward=False, grip_in_action=True,
                    default_grip_state='o', num_objs=1)
env.seed(0)
env.render()
obs = env.reset()

for i in range(num_steps):
    act = env.action_space.sample()
    next_obs, rew, done, info = env.step(act)

    # env.render()
    obs = next_obs