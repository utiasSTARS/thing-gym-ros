import sys
import time

import gym
import numpy as np

from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv
from thing_gym_ros.envs.reaching.visual import ThingRosReachAndGraspXYImageMB

np.set_printoptions(suppress=True, precision=4)


# params
num_steps = 50
gap_between_actions = 4

# vars
all_obs = []
all_act = []
next_new_action = 0

# env = ThingRosMBEnv(img_in_state=True, depth_in_state=True, dense_reward=False, grip_in_action=True,
#                     default_grip_state='o', num_objs=1,
#                     state_data=('pose', 'grip_pos', 'prev_grip_pos'))
env = ThingRosReachAndGraspXYImageMB(False, True)

env.seed(0)
env.render()
obs = env.reset()

for i in range(num_steps):
    if next_new_action == 0:
        act = env.action_space.sample() * .1
        next_new_action = gap_between_actions
    next_new_action -= 1

    act_and_env_start = time.time()

    # testing "processing" time
    # time.sleep(.05)

    # act = [0.005, 0.0, 0.0, 0.0, 0, 0, -1.0]
    env_start = time.time()
    next_obs, rew, done, info = env.step(act)
    # print('env time: ', time.time() - env_start)
    # print('total time: ', time.time() - act_and_env_start)
    all_act.append(act)
    all_obs.append(obs)

    # env.render()
    obs = next_obs