# built-ins
import sys
import os
import copy
from multiprocessing import Lock

# ros
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
import tf2_ros
import rostopic
from std_msgs.msg import Float64MultiArray, Bool

# other
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
import pygame
import yaml

# local
import ros_np_tools as rnt
from thing_gym_ros.msg import KeyboardTrigger


class ThingRosEnv(gym.Env):
    # todo implement as needed
    # CONTROL_TYPES = ('delta_tool, delta_joint, pos_tool, pos_joint, vel_tool, vel_joint')
    CONTROL_TYPES = ('delta_tool')
    CONTROL_TYPES_WITH_QUAT = ('delta_tool, pos_tool')

    def __init__(self,
                 img_in_state,
                 depth_in_state,
                 dense_reward,
                 grip_in_action,
                 default_grip_state,  # 'o' for open, 'c' for closed
                 num_objs,  # number of objects that can be interacted with
                 robot_config_file=None,  # yaml config file
                 state_data=('pose', 'prev_pose', 'grip_pos', 'obj_pos', 'obj_rot'),
                 valid_act_t_dof=(1, 1, 1),
                 valid_act_r_dof=(1, 1, 1),
                 num_prev_pos=5,
                 gap_between_prev_pos=.1,  # in seconds
                 moving_base=False,  # whether base moves to different views between episodes
                 max_real_time=5,  # in seconds
                 success_causes_done=False,
                 failure_causes_done=False
                 ):
        """
        Requires thing + related topics and action servers (either in sim or reality) to already be launched
        separately.

        Quaternions for actions should be entered as XYZW to match ROS's formatting.
        """

        rospy.init_node('thing_gym')
        self.sim = rospy.get_param('/simulation')

        # load config
        base_config_file = 'configs/base.yaml' if not self.sim else 'configs/base_sim.yaml'
        with open(base_config_file) as f:
            cfg = yaml.load(f)

        if robot_config_file is not None:
            with open(robot_config_file) as f:
                new_config = yaml.load(f)
            for k in new_config:
                cfg[k] = new_config[k]

        # env setup
        assert cfg['control_type'] in ThingRosEnv.CONTROL_TYPES, '%s is not in the valid control types %s' % \
                                                          (cfg['control_type'], ThingRosEnv.CONTROL_TYPES)
        self._control_type = cfg['control_type']
        self.grip_in_action = grip_in_action
        self.default_grip_state = default_grip_state
        self.state_data = state_data
        self.img_in_state = 'img_in_state'
        self.depth_in_state = 'depth_in_state'
        self.image_width = cfg['image_width']
        self.image_height = cfg['image_height']
        self.image_zoom_crop = cfg['image_zoom_crop']
        self._control_freq = cfg['control_freq']
        self._max_episode_steps = max_real_time * self._control_freq
        self.valid_act_t_dof = valid_act_t_dof
        self.valid_act_r_dof = valid_act_r_dof
        self.pos_limits = cfg['pos_limits']
        self.arm_max_trans_vel = cfg['arm_max_trans_vel']
        self.arm_max_rot_vel = cfg['arm_max_rot_vel']
        if self._control_type in ThingRosEnv.CONTROL_TYPES_WITH_QUAT:
            self._quat_in_action = True
        else:
            self._quat_in_action = False

        # gym setup
        self._num_trans = sum(self.valid_act_t_dof)
        if self.valid_act_r_dof > 0 and self._quat_in_action:
            self._num_rot = 4  # for quat
        else:
            self._num_rot = sum(self.valid_act_r_dof)
        self._valid_len = self._num_trans + self._num_rot + self.grip_in_action

        self.action_space = spaces.Box(-1, 1, (self._valid_len,), dtype=np.float32)
        state_size = ('pose' in self.state_data) * 7 + \
                     ('prev_pose' in self.state_data) * 7 * num_prev_pos + \
                     ('grip pos' in self.state_data) * 2 + \
                     ('obj_pos' in self.state_data) * 3 * num_objs + \
                     ('obj_rot' in self.state_data) * 4 * num_objs + \
                     ('obj_rot_z' in self.state_data or 'obj_rot_z_90' in self.state_data or
                      'obj_rot_z_180' in self.state_data) * 2 * num_objs
        state_space = spaces.Box(-np.inf, np.inf, (state_size,), dtype=np.float32)
        if img_in_state or depth_in_state:
            obs_space_dict = dict()
            obs_space_dict['obs'] = state_space
            if img_in_state:
                obs_space_dict['img'] = spaces.Box(0, 255, (self.image_height, self.image_width, 3), dtype=np.uint8)
            if depth_in_state:
                obs_space_dict['depth'] = spaces.Box(0, 1, (self.image_height, self.image_width), dtype=np.float32)
            self.observation_space = spaces.Dict(spaces=obs_space_dict)
        else:
            self.observation_space = state_space

        # hard-coded parameter from thing_control -- time_from_start in servoCallback
        # should be set to the same value as control freq, ideally
        self.__action_duration = .1

        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # tf objs for updating poses
        self.tf_odom_base = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'base_link')
        self.tf_base_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'thing_tool')

        # subscribers
        if img_in_state:  # defaults are based on using a kinect
            if cfg['img_topic'] is None:
                img_topic = '/camera/rgb/image_raw' if self.sim else '/camera/sd/image_color_rect'
            else:
                img_topic = cfg['img_topic']
            self.sub_img = rospy.Subscriber(img_topic, Image, self.img_cb)
            self.img_lock = Lock()
        if depth_in_state:
            if cfg['depth_topic'] is None:
                depth_topic = '/camera/depth/image_raw' if self.sim else '/camera/sd/image_depth_rect'
            else:
                depth_topic = cfg['depth_topic']
            self.sub_depth = rospy.Subscriber(depth_topic, Image, self.depth_cb)
            self.depth_lock = Lock()

        # publishers
        self.pub_servo = rospy.Publisher('/servo/command', Float64MultiArray, queue_size=1)
        self.pub_gripper = rospy.Publisher('FRL/remote_trigger', KeyboardTrigger, queue_size=10)

        # TODO confirm that if there is a sleep command in the step call, the subscribers
        # will still update their data buffers
        # TODO confirm logic here: if we sleep with rospy Rate, it's possible that the time between
        # an action and the observation will not be consistent between loops, and this may be undesireable,
        # particularly if we're trying to write code for predicting future states
        self.rate = rospy.Rate(self._control_freq)

    def set_max_episode_steps(self, n):
        self._max_episode_steps = n

    def seed(self, seed=None):
        """ Seed for random numbers, for e.g. resetting the environment """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ If action space requires a quat, the quat does not have to be entered normalized to be valid.

        Action should come in as (n,) shape array, where n includes number of translational DOF, +4 if any rotational
        DOF (for quats), and +1 if gripper control is included. Gripper control is a float where anything below 0
        is considered open, and anything above 0 is considered close.
        """
        assert len(action) == self._valid_len, 'action needs %d dimensions for this env, step called with %d' % \
                                         (self._valid_len, len(action))

        # update current pose data from tf trees
        self.tf_base_tool.update()
        self.tf_odom_base.update()

        # process and publish robot action
        if self._control_type == 'delta_tool':
            delta_trans = np.array([0., 0., 0.])
            delta_trans[self.valid_act_t_dof.nonzero()[0]] = action[:num_trans]
            delta_rot = np.array(action[num_trans:(num_trans + num_rot)])

            action[:2], limit_reached = self._limit_action()


        # process and publish grip action
        if self.grip_in_action:
            if self.default_grip_state == 'o':
                grip = 'c' if action[-1] > 0 else 'o'
            else:
                grip = 'o' if action[-1] > 0 else 'c'
        else:
            grip = self.default_grip_state
        g_msg = KeyboardTrigger()
        g_msg.label = grip
        self.pub_gripper.publish(g_msg)

        # specifically enforce duration to be constant before observation, rospy.rate.sleep wouldn't
        # allow this
        rospy.sleep(1 / self._control_freq)

        # get observation

        # get reward

    def _limit_action(self, action):
        """ Limit the desired action based on pos and vel maximums. """


    def reset(self):



class Talker():
    # just the regular python ros tutorial node
    def __init__(self):

        print('Version info: ', sys.version_info)

        pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            rospy.loginfo(hello_str)
            pub.publish(hello_str)
            rate.sleep()