# built-ins
import sys
import os
import copy
import time
from threading import Thread, Lock
import queue
from queue import Queue

# ros
import rospy
from geometry_msgs.msg import Transform, TransformStamped, WrenchStamped
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Path
import tf2_ros
import tf.transformations as tf_trans
import rostopic
from std_msgs.msg import Float64MultiArray, Bool
from cv_bridge import CvBridge, CvBridgeError
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionFeedback
import dynamic_reconfigure.client
# from visualization_msgs.msg import Marker
from control_msgs.msg import JointTrajectoryControllerState

# other
import numpy as np
from numpy.linalg import norm
import gym
from gym.utils import seeding
from gym import spaces
import yaml
import cv2

# local
import ros_np_tools as rnt
import thing_gym_ros.envs.utils as thing_gym_ros_env_utils
from thing_gym_ros_catkin.msg import KeyboardTrigger
from thing_gym_ros_catkin.msg import SModel_robot_input, SModel_robot_output  # copied directly from Robotiq package
from thing_gym_ros_catkin.msg import Marker  # need the indigo version to communicate with thing rviz


XYZ_DEFAULTS = dict(valid_act_t_dof=(1, 1, 1), valid_act_r_dof=(0, 0, 0))
SIXDOF_DEFAULTS = dict(valid_act_t_dof=(1, 1, 1), valid_act_r_dof=(1, 1, 1))


class ThingRosEnv(gym.Env):
    # implement as needed
    # CONTROL_TYPES = ('delta_tool, delta_joint, pos_tool, pos_joint, vel_tool, vel_joint')
    CONTROL_TYPES = ('delta_tool')

    def __init__(self,
                 img_in_state,
                 depth_in_state,
                 dense_reward,
                 grip_in_action,
                 default_grip_state,  # 'o' for open, 'c' for closed
                 num_objs,  # number of objects that can be interacted with
                 robot_config_file=None,  # yaml config file
                 state_data=('pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'force_torque',
                             'timestep'),
                 valid_act_t_dof=(1, 1, 1),
                 valid_act_r_dof=(1, 1, 1),
                 max_real_time=5,  # in seconds
                 success_causes_done=False,
                 failure_causes_done=False,
                 reset_teleop_available=False,
                 success_feedback_available=False,
                 num_grip_fingers=3,
                 position_impedance_control=False,
                 init_gripper_random_lim=(0, 0, 0, 0, 0, 0),
                 info_env_only=False,  # if True, env can only be used for getting space sizes and other info
                 high_ft_causes_failure=False,
                 high_ft_causes_failure_thresh=[80, 22]
                 ):
        """
        Requires thing + related topics and action servers (either in sim or reality) to already be launched
        separately.
        """

        if not info_env_only:
            rospy.init_node('thing_gym')
            self.sim = rospy.get_param('/simulation')
        else:
            self.sim = False

        # load config
        base_config_file = 'configs/base.yaml' if not self.sim else 'configs/base_sim.yaml'
        base_config_file = os.path.join(os.path.dirname(__file__), base_config_file)
        with open(base_config_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        if robot_config_file is not None:
            with open(robot_config_file) as f:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            for k in new_config:
                self.cfg[k] = new_config[k]

        # env setup
        assert self.cfg['control_type'] in ThingRosEnv.CONTROL_TYPES, '%s is not in the valid control types %s' % \
                                                                      (self.cfg['control_type'],
                                                                       ThingRosEnv.CONTROL_TYPES)
        self._control_type = self.cfg['control_type']
        self._poses_ref_frame = self.cfg['poses_ref_frame']
        self._rot_act_rep = self.cfg['rot_act_rep']
        self.img_in_state = img_in_state
        self.depth_in_state = depth_in_state
        self.success_causes_done = success_causes_done
        self.failure_causes_done = failure_causes_done
        self.done_success = False
        self.done_failure = False
        self.done_timeout = False
        self.dense_reward = dense_reward
        self.num_objs = num_objs
        self.position_impedance_control = position_impedance_control
        self.init_gripper_random_lim = init_gripper_random_lim
        self.pic_K_trans = self.cfg['position_impedance_K_trans']
        self.pic_D_trans = self.cfg['position_impedance_D_trans']
        self.pic_K_rot = self.cfg['position_impedance_K_rot']
        self.pic_D_rot = self.cfg['position_impedance_D_rot']
        self.pic_max_F = self.cfg['position_impedance_max_F']
        self.pic_max_T = self.cfg['position_impedance_max_T']
        self.image_width = self.cfg['img_width']
        self.image_height = self.cfg['img_height']
        self.image_center_crop = self.cfg['img_center_crop']
        self.image_crop = self.cfg['img_crop']
        self._control_freq = self.cfg['control_freq']
        self.action_lpf_beta = self.cfg['action_lpf_beta']
        self._max_episode_steps = int(max_real_time * self._control_freq)

        self.num_prev_pose = self.cfg['num_prev_pose']
        self.num_prev_grip = self.cfg['num_prev_grip']
        self.pos_limits = self.cfg['pos_limits']
        self.arm_max_trans_vel = self.cfg['arm_max_trans_vel']
        self.arm_max_rot_vel = self.cfg['arm_max_rot_vel']
        self._moving_base = False  # overwritten by moving base child class
        self._max_depth = self.cfg['depth_max_dist']
        self._require_img_depth_registration = self.cfg['require_img_depth_registration']
        self._sensor = self.cfg['sensor']
        self._cam_forward_axis = self.cfg['cam_forward_axis']
        self.state_data = state_data
        self.valid_act_t_dof = np.array(valid_act_t_dof)
        self.valid_act_r_dof = np.array(valid_act_r_dof)
        self.grip_in_action = grip_in_action
        self.default_grip_state = default_grip_state
        if self._rot_act_rep == 'quat':
            self._quat_in_action = True
            raise NotImplementedError('Implement if needed')
        else:
            self._quat_in_action = False
        self.high_ft_causes_failure = high_ft_causes_failure
        self.high_ft_causes_failure_thresh = high_ft_causes_failure_thresh

        # gym setup
        self._num_trans = sum(self.valid_act_t_dof)
        if sum(self.valid_act_r_dof) > 0 and self._quat_in_action:
            self._num_rot = 4  # for quat
        else:
            self._num_rot = sum(self.valid_act_r_dof)
        self._valid_len = self._num_trans + self._num_rot + self.grip_in_action

        self.action_space = spaces.Box(-1, 1, (self._valid_len,), dtype=np.float32)
        state_size = ('pose' in self.state_data) * 7 + \
                     ('prev_pose' in self.state_data) * 7 * self.num_prev_pose + \
                     ('grip_pos' in self.state_data) * num_grip_fingers + \
                     ('prev_grip_pos' in self.state_data) * num_grip_fingers * self.num_prev_grip + \
                     ('obj_pos' in self.state_data) * 3 * num_objs + \
                     ('obj_rot' in self.state_data) * 4 * num_objs + \
                     ('obj_rot_z' in self.state_data or 'obj_rot_z_90' in self.state_data or
                      'obj_rot_z_180' in self.state_data) * 2 * num_objs + \
                     ('force_torque' in self.state_data) * 6 + \
                     ('timestep' in self.state_data)

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

        if info_env_only:
            return

        # marker in rviz for pos limits
        self.pub_pos_limits = rospy.Publisher('pos_limits_marker', Marker, queue_size=1)
        self.pos_limits_marker = Marker()
        self.pos_limits_marker.header.frame_id = 'base_link'
        self.pos_limits_marker.header.stamp = rospy.Time(0)
        self.pos_limits_marker.ns = 'pos_limits'
        self.pos_limits_marker.id = 0
        self.pos_limits_marker.type = Marker.CUBE
        self.pos_limits_marker.action = Marker.ADD
        self.pos_limits_marker.pose.position.x = (self.pos_limits[0] + self.pos_limits[3]) / 2
        self.pos_limits_marker.pose.position.y = (self.pos_limits[1] + self.pos_limits[4]) / 2
        self.pos_limits_marker.pose.position.z = (self.pos_limits[2] + self.pos_limits[5]) / 2
        self.pos_limits_marker.pose.orientation.w = 1.0  # xyz default to 0
        self.pos_limits_marker.scale.x = np.abs(self.pos_limits[0] - self.pos_limits[3])
        self.pos_limits_marker.scale.y = np.abs(self.pos_limits[1] - self.pos_limits[4])
        self.pos_limits_marker.scale.z = np.abs(self.pos_limits[2] - self.pos_limits[5])
        self.pos_limits_marker.color.g = 1.0
        self.pos_limits_marker.color.a = .3
        self.pos_limits_marker.lifetime = rospy.Duration()

        # parameters for thing_control
        self.thing_control_dyn_rec_client = dynamic_reconfigure.client.Client('/thing_control')
        self._action_duration = self.cfg['servo_time_from_start']
        self._time_between_poses_tc = self.cfg['traj_time_between_points']
        params_dict = {'servo_time_from_start': self.cfg['servo_time_from_start'],
                       'traj_time_between_points': self.cfg['traj_time_between_points']}
        print("Updating dynamic reconfigure parameters in thing_control")
        self.thing_control_dyn_rec_client.update_configuration(params_dict)

        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # tf objs for updating poses
        self.tf_odom_base = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'base_link')
        self.tf_base_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'thing_tool')
        self.tf_odom_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'thing_tool')

        # publishers
        self.pub_servo = rospy.Publisher('/servo/command', Float64MultiArray, queue_size=1)
        self.pub_gripper = rospy.Publisher('FRL/remote_trigger', KeyboardTrigger, queue_size=10)
        self.pub_gripper_raw = rospy.Publisher('/SModelRobotOutput', SModel_robot_output, queue_size=1)
        self.pub_base_traj = rospy.Publisher('goal_ridgeback', Path, queue_size=1)
        self.pub_arm_traj = rospy.Publisher('goal_ur10', Path, queue_size=1)

        # using both a fixed sleep and variable sleep to match env with and without processing time on obs
        self.rate = rospy.Rate(self._control_freq)
        assert self.cfg['max_policy_time'] < 1 / self._control_freq, "max_policy_time %.3f is not less than period " \
                                    "defined by control_freq %.3f" % (self.cfg['max_policy_time'], 1 / self._control_freq)
        self._max_policy_time = self.cfg['max_policy_time']
        self._fixed_time_after_action = 1 / self._control_freq - self._max_policy_time

        # resetting thresholds and parameters
        self._reset_base_tool_tf_arr = np.array(self.cfg['reset_base_tool_tf'])
        self._reset_base_tool_mat= tf_trans.euler_matrix(*self._reset_base_tool_tf_arr[3:])
        self._reset_base_tool_mat[:3, 3] = self._reset_base_tool_tf_arr[:3]
        self._reset_joint_pos = np.array(self.cfg['reset_joint_pos'])
        self._max_reset_trans = 2.0  # meters
        self._max_reset_rot = 2.5  # radians
        self._reset_vel_trans = .15  # m/s
        self._reset_vel_rot = .5  # rad/s
        self._reset_base_vel_trans = .15  # m/s
        self._reset_base_vel_rot = .3  # rad/s
        self._reset_teleop_available = reset_teleop_available
        if reset_teleop_available:
            self.reset_teleop_complete = False
        self._success_feedback_available = success_feedback_available

        # gui
        self.gui_thread = None
        self.env_to_gui_q = None
        self.gui_to_env_q = None
        self.gui_timer = None
        self.gui_dict = None
        self.gui_lock = Lock()
        self.play_pause_env = True
        self.latest_processed_img = None

        # other attributes
        self.prev_action = None
        self.prev_pose = None
        self.prev_grip_pos = None
        self.cv_bridge = CvBridge()
        self.ep_timesteps = 0
        self._img_depth_registered = None
        self.ep_odom_base_mat = self.tf_odom_base.as_mat()
        self._env_reset_complete = False

        # subscribers
        if img_in_state:  # defaults are based on using a kinect
            if self.cfg['alt_img_topic'] is None:
                img_topic = '/camera/rgb/image_raw' if self.sim else '/camera/sd/image_color_rect'
            else:
                img_topic = self.cfg['alt_img_topic']
            self.sub_img = rospy.Subscriber(img_topic, Image, self.img_cb)
            self.img_lock = Lock()
            self.latest_img = None
            self._raw_img_height, self._raw_img_width = None, None
        if depth_in_state:
            if self.cfg['alt_depth_topic'] is None:
                depth_topic = '/camera/depth/image_raw' if self.sim else '/camera/sd/image_depth_rect'
            else:
                depth_topic = self.cfg['alt_depth_topic']
            self.sub_depth = rospy.Subscriber(depth_topic, Image, self.depth_cb)
            self.depth_lock = Lock()
            self.latest_depth = None
            self._raw_depth_height, self._raw_depth_width = None, None

        if 'force_torque' in self.state_data or self.position_impedance_control:
            self.sub_ft = rospy.Subscriber('/robotiq_force_torque_wrench_zero', WrenchStamped, self.ft_cb)
            self.pub_ft_zero = rospy.Publisher('/FT_sensor_bias_node/set_zero', Bool, queue_size=1)
            self.ft_lock = Lock()
            self.ft_filter_size = 5
            self.ft_filter = None
            self.latest_ft = None
            self.latest_ft_raw = None
            if self._poses_ref_frame == 'b':
                self.tf_ft_ref = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'FT300_link', 'base_link')
            elif self._poses_ref_frame == 'w':
                self.tf_ft_ref = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'FT300_link', 'odom')
            self.tf_ft_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'FT300_link', 'thing_tool')

        if grip_in_action:
            self.sub_grip = rospy.Subscriber('/SModelRobotInput', SModel_robot_input, self.grip_cb)
            self.grip_lock = Lock()
            self.latest_grip = None
            self.latest_grip_bool = None

        if 'grip_pos' in self.state_data or 'prev_grip_pos' in self.state_data:
            assert grip_in_action, "Env created with grip_pos in state data but with grip_in_action set to False"

        # subscriber for joint states
        self.arm_joint_states_sub = rospy.Subscriber('/vel_based_pos_traj_controller/state',
                                                     JointTrajectoryControllerState, self.arm_joint_states_cb)
        self.arm_joint_pos_lock = Lock()

        rospy.sleep(1.0)  # allow publishers to get ready
        self.pub_pos_limits.publish(self.pos_limits_marker)

        if self.grip_in_action:
            rospy.set_param('/gripper_force', self.cfg['max_grip_force'])
            # grip_force_msg = SModel_robot_output()
            # grip_force_msg.rFRA = self.cfg['max_grip_force']
            # self.pub_gripper.publish(grip_force_msg)

    def set_max_episode_steps(self, n):
        self._max_episode_steps = n

    def seed(self, seed=None):
        """ Seed for random numbers, for e.g. resetting the environment """
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def step(self, action, reset_teleop_step=False):
        """ If action space requires a quat, the quat does not have to be entered normalized to be valid.

        Action should come in as (n,) shape array, where n includes number of translational DOF, rotational DOF,
        and +1 if gripper control is included. Gripper control is a float where anything below 0
        is considered open, and anything above 0 is considered close.
        """
        assert len(action) == self._valid_len, 'action needs %d dimensions for this env, step called with %d' % \
                                         (self._valid_len, len(action))

        if not reset_teleop_step and not self._env_reset_complete:
            raise RuntimeError("Call to reset did not complete, possibly because teleop is available and "
                               "set_reset_teleop_complete was not called.")

        if self.action_lpf_beta < 1 and self.prev_action is not None:
            action = self.action_lpf_beta * action + (1 - self.action_lpf_beta) * self.prev_action

        # gui handling
        if self.gui_thread is not None:
            self.gui_lock.acquire()
            if not self.play_pause_env:
                self.gui_lock.release()
                print("Env is paused, unpause using gui.")
                self.gui_lock.acquire()
                while not self.play_pause_env:
                    self.gui_lock.release()
                    rospy.sleep(.1)
                    self.gui_lock.acquire()
            self.gui_lock.release()

        self.gui_lock.acquire()

        # update current pose data from tf trees
        self.tf_odom_tool.update()
        self.tf_base_tool.update()
        self.tf_odom_base.update()

        # process and publish robot action
        if self._control_type == 'delta_tool':
            delta_trans = np.array([0., 0., 0.])
            delta_trans[self.valid_act_t_dof.nonzero()[0]] = action[:self._num_trans]
            T_delta_trans = np.eye(4)
            T_delta_trans[:3, 3] = delta_trans
            rod_delta_rot = np.array([0., 0., 0.])
            rod_delta_rot[self.valid_act_r_dof.nonzero()[0]] = action[self._num_trans:(self._num_trans + self._num_rot)]
            ang = norm(rod_delta_rot)
            eps = 1e-8
            if ang < eps:
                ang = 0
                axis = np.array([1, 0, 0])
            else:
                axis = rod_delta_rot / ang
            R_delta_rot = tf_trans.rotation_matrix(ang, axis)[:3, :3]

            # we want movement to happen starting from the current tool position, but using the axes of the
            # current ref frame
            T_odom_tool = self.tf_odom_tool.as_mat()

            T_act_frame = copy.deepcopy(T_odom_tool)
            if self._poses_ref_frame == 'b':
                # T_base_tool = self.tf_base_tool.as_mat()
                T_odom_base = self.tf_odom_base.as_mat()
                T_act_frame[:3, :3] = T_odom_base[:3, :3]
                R_ref_tool = self.tf_base_tool.as_mat()[:3, :3]
                R_odom_ref = T_odom_base[:3, :3]
            elif self._poses_ref_frame == 'w':
                T_act_frame[:3, :3] = np.eye(3)
                R_ref_tool = T_odom_tool[:3, :3]
                R_odom_ref = np.eye(3)
            # change on pos is based on transforming from ee point but along ref axes
            T_new = T_act_frame.dot(T_delta_trans)
            # change in rot is based on modifying ref to tool rotation, based on ref rot, and then putting it
            # in the odom frame
            T_new[:3, :3] = R_odom_ref.dot(R_delta_rot.dot(R_ref_tool))

            T_new, limit_reached = self._limit_action(T_new)

            if self.position_impedance_control:
                np.set_printoptions(suppress=True, precision=4)
                # print("Before fix: ", T_new[:3, 3])
                T_new = self._position_impedance_control_action(T_new)
                # print("After fix: ", T_new[:3, 3])

            servo_msg = rnt.thing.get_servo_msg(mat=T_new, base_tf_msg=rnt.tf_msg.mat_to_tf_msg(self.ep_odom_base_mat))

        # process grip action
        if self.grip_in_action:
            if self.default_grip_state == 'o':
                grip = 'c' if action[-1] > 0 else 'o'
            else:
                grip = 'o' if action[-1] > 0 else 'c'
        else:
            grip = self.default_grip_state
        g_msg = KeyboardTrigger()

        # modify gripper based on config for max gripper open pos
        # if grip == 'o':
        #     grip = '75'

        g_msg.label = grip

        self.pub_servo.publish(servo_msg)
        self.pub_gripper.publish(g_msg)

        self.gui_lock.release()
        rospy.sleep(self._fixed_time_after_action)
        self.gui_lock.acquire()

        # get and process observation
        obs, full_obs_dict = self._prepare_obs()

        self.gui_lock.release()
        self.rate.sleep()
        self.gui_lock.acquire()

        # get reward
        r = self._get_reward()

        # get done
        if not reset_teleop_step:
            self.ep_timesteps += 1

            # get high ft causing failure
            if self.high_ft_causes_failure:
                if not self.done_failure:
                    self.done_failure = self._get_high_ft_failure()
            # print("Force norm: ", norm(self.latest_ft_raw[:3]), ", Torque norm: ", norm(self.latest_ft_raw[3:]))

            done = self._get_done()

            # info includes information about success and failure
            info = dict(done_success=False, done_failure=False)
            if done:
                # get feedback on success only if a timeout ocurred
                if self.done_timeout:
                    if self._success_feedback_available:
                        self.gui_lock.release()
                        if self._reset_teleop_available:
                            print("Waiting for user feedback on success: press up success, down for fail. "
                                  "This must be taken care of in code handling teleop.")
                            user_success_feedback = False
                        else:
                            user_success_feedback = input("Waiting for user feedback on success: press s then enter for success, "
                                                          "or just enter for failure.")
                        self.gui_lock.acquire()
                        if user_success_feedback == 's':
                            info['done_success'] = True
                elif self.done_success:
                    info['done_success'] = True
                elif self.done_failure:
                    info['done_failure'] = True
        else:
            done = False
            info = {}

        self.prev_action = action
        self.gui_lock.release()
        return obs, r, done, info

    def _limit_action(self, action):
        """ Limit the desired action based on pos and vel maximums. """
        limit_reached = False
        self.tf_odom_base.update()
        self.tf_base_tool.update()
        self.tf_odom_tool.update()

        if self._control_type == 'delta_tool':
            T_des_odom_tool = action
            if self._poses_ref_frame == 'b':
                # assuming that pos limits are given relative to base
                T_odom_base = self.tf_odom_base.as_mat()
                T_base_odom = rnt.tf_mat.invert_transform(T_odom_base)
                T_des_base_tool = T_base_odom.dot(T_des_odom_tool)
                if np.any(T_des_base_tool[:3, 3] < self.pos_limits[:3]) or \
                    np.any(T_des_base_tool[:3, 3] > self.pos_limits[3:]):
                    limit_reached = True
                T_des_base_tool[:3, 3] = np.clip(T_des_base_tool[:3, 3], self.pos_limits[:3], self.pos_limits[3:])
                limited_action = T_odom_base.dot(T_des_base_tool)
            elif self._poses_ref_frame == 'w':
                # assuming pos limits given relative to odom
                if np.any(T_des_base_tool[:3, 3] < self.pos_limits[:3]) or \
                    np.any(T_des_base_tool[:3, 3] > self.pos_limits[3:]):
                    limit_reached = True
                limited_action = np.clip(T_des_odom_tool[:3, 3], self.pos_limits[:3], self.pos_limits[3:])

        else:
            raise NotImplementedError('No limits implemented for chosen control type.')

        return limited_action, limit_reached

    def _position_impedance_control_action(self, T_des):
        """ Return a new action based on simple position impedance. """
        K = np.diag([self.pic_K_trans] * 3 + [self.pic_K_rot] * 3)
        D = np.diag([self.pic_D_trans] * 3 + [self.pic_D_rot] * 3)
        ft = self.latest_ft_raw
        self.tf_ft_ref.update()
        T_ft_ref_to_des_ref = self.tf_ft_ref.as_mat()
        if self._poses_ref_frame == 'w':
            T_tool_to_des = rnt.tf_mat.invert_transform(self.tf_odom_tool.as_mat()).dot(T_des)
        elif self._poses_ref_frame == 'b':
            T_tool_to_des = rnt.tf_mat.invert_transform(self.tf_base_tool.as_mat()).dot(T_des)
        f_max = self.pic_max_F
        t_max = self.pic_max_T

        # get current velocity for damping
        if self.prev_pose is not None:
            vel_t = self.prev_pose[0, :3] - self.prev_pose[1, :3]
            q_diff = tf_trans.quaternion_multiply(tf_trans.quaternion_inverse(self.prev_pose[1, 3:]),
                                                  self.prev_pose[0, 3:])
            ang, ax, _ = tf_trans.rotation_from_matrix(tf_trans.quaternion_matrix(q_diff))
            vel_r = ax * ang
            vel = np.concatenate([vel_t, vel_r])
        else:
            vel = np.zeros([6])

        T_mod = rnt.thing.get_position_impedance_control_action(T_des, K, D, ft, f_max, t_max, T_ft_ref_to_des_ref,
                                                                self.tf_ft_tool.as_mat(), T_tool_to_des, vel)
        return T_mod

    def _get_reward(self):
        """ Should be overwritten by children. """
        return 0.0

    def _get_high_ft_failure(self):
        """ Gives True if the latest high ft is above a threshold, to be used as a failure condition, ideally
        to prevent the robot from protective stopping. """
        ft = self.latest_ft_raw
        f_norm = norm(ft[:3])
        t_norm = norm(ft[3:])
        thresh = self.high_ft_causes_failure_thresh
        if f_norm > thresh[0]:
            print("Episode failure caused by high force of %.3f, threshold %.3f" % (f_norm, thresh[0]))
            return True
        elif t_norm > thresh[1]:
            print("Episode failure caused by high torque of %.3f, threshold %.3f" % (t_norm, thresh[1]))
            return True
        else:
            return False

    def _get_done(self):
        """ Can be overwritten by children, but this gives a default based on human feedback or timeout. """
        if self.ep_timesteps == self._max_episode_steps:
            self.done_timeout = True
            return True
        elif (self.success_causes_done or self.failure_causes_done) and \
            (self.done_failure or self.done_success):
            return True
        else:
            return False

    def reset(self):
        """ Reset the environment to the beginning of an episode.

        In sim, a user could theoretically reload or otherwise move objects arbitrarily, but since the
        primary use for this class is for the real robot, this method will require interaction with a person."""

        self.gui_lock.acquire()
        self._env_reset_complete = False

        # For convenience, this allows a user to reset the environment using their teleop and move the EE
        # into an ideal pose to then be driven back to the initial pose.
        # A calling program resetting the environment would then do the following:
        #
        # env.reset() --> given output from _reset_help, user teleops robot to reset env
        # while not teleop_button_to_indicate_reset_done:
        #     action = get_teleop_action(current_env_pose)
        #     env.step(action)
        # env.set_reset_teleop_complete()
        # env.reset()
        if self._reset_teleop_available and not self.reset_teleop_complete:
            print("reset called with teleop available. Reset objects to new poses given by helper, "
                  "then calling program calls set_reset_teleop_complete and reset again")
            self._reset_helper()
            self.gui_lock.release()
            return

        # randomize initial gripper pose
        if self.init_gripper_random_lim != (0, 0, 0, 0, 0, 0):
            ep_reset_base_tool_tf_arr = self._reset_base_tool_tf_arr + self.np_random.uniform(
                low=-np.array(self.init_gripper_random_lim) / 2,
                high=np.array(self.init_gripper_random_lim) / 2, size=6)
            # euler is a bad idea for large sample space, but for small should be fine
            self._reset_base_tool_mat = tf_trans.euler_matrix(*ep_reset_base_tool_tf_arr[3:])
            self._reset_base_tool_mat[:3, 3] = ep_reset_base_tool_tf_arr[:3]

        # first do safety checks to make sure movement isn't too dramatic
        self.tf_base_tool.update()
        base_tool_pos_quat = self.tf_base_tool.as_pos_quat()
        dist_arm_to_init, rot_dist_arm_to_init = rnt.pos_quat_np.get_trans_rot_dist(base_tool_pos_quat,
                                                 rnt.pos_quat_np.mat_to_pos_quat(self._reset_base_tool_mat))
        # dist_arm_to_init = norm(base_tool_pos_quat[:3] - self._reset_base_tool_mat[3, :3])
        if dist_arm_to_init > self._max_reset_trans:
            raise RuntimeError("EE is %.3fm from initial pose. Must be within %.3fm to reset." %
                               (dist_arm_to_init, self._max_reset_trans))
            # input("Move arm closer to initial pose and press Enter to continue...")
            # self.tf_base_tool.update()
            # dist_arm_to_init = norm(base_tool_pos_quat[:3] - self._reset_base_tool_mat[3, :3])
        # rot_dist_arm_to_init = np.arccos(
        #     np.clip((np.trace(np.dot(base_tool_mat[:3, :3], self._reset_base_tool_mat[:3, :3].T)) - 1) / 2, -1.0, 1.0))
        if rot_dist_arm_to_init > self._max_reset_rot:
            raise RuntimeError("EE is %.3frad from init pose. Must be within %.3frad." %
                               (rot_dist_arm_to_init, self._max_reset_rot))
            # input("Move arm closer to initial pose and press Enter to continue...")
            # self.tf_base_tool.update()
            # rot_dist_arm_to_init = np.arccos(
            #     np.clip((np.trace(np.dot(base_tool_mat[:3, :3], self._reset_base_tool_mat[:3, :3].T)) - 1) / 2, -1.0,
            #             1.0))

        # complete the movement -- thing_control takes all motions in the frame of odom
        if not self._reset_teleop_available:
            self.gui_lock.release()
            input("Ensure there is a linear, collision-free path between end-effector and initial pose,"
                  "then press Enter to continue...")
            self.gui_lock.acquire()

        self._publish_reset_trajectories()

        # Need reset trajectories to completely finish before grabbing observation and returning
        # action client is super buggy, so the right way to do this would be using the
        # follow_joint_trajectory/result topics, but instead we'll just use a simple poll
        reset_timeout_republish_time = 5.5  # in case actionlib barfs -- should be longer than thing_control wait time
        reset_non_movement_republish_time = 5.5  # again, for actionlib errors
        reset_thresh_trans = .01
        reset_thresh_rot = .1
        settle_time = 1.5
        time_within_spec = 0.0
        within_spec = False
        both_started = False
        within_spec_start = 0
        pub_start_time = time.time()
        dist_base_to_reset, rot_dist_base_to_reset = rnt.pos_quat_np.get_trans_rot_dist(
            self.tf_odom_base.as_pos_quat(), rnt.pos_quat_np.mat_to_pos_quat(self.ep_odom_base_mat))
        init_dist_base_to_reset = dist_base_to_reset
        init_rot_dist_base_to_reset = rot_dist_base_to_reset
        init_dist_arm_to_init = dist_arm_to_init
        init_rot_dist_arm_to_init = rot_dist_arm_to_init
        while time_within_spec < settle_time:
            cur_time = time.time()
            if (dist_arm_to_init < reset_thresh_trans and rot_dist_arm_to_init < reset_thresh_rot and
               dist_base_to_reset < reset_thresh_trans and rot_dist_base_to_reset < reset_thresh_rot):
                if not within_spec:
                    within_spec = True
                    within_spec_start = time.time()
            else:
                within_spec = False
            if within_spec:
                time_within_spec = time.time() - within_spec_start
            else:
                time_within_spec = 0.0
            if norm(init_dist_base_to_reset - dist_base_to_reset) > .01 and \
                norm(init_dist_arm_to_init - dist_arm_to_init) > .01:
                both_started = True
            if cur_time - pub_start_time > reset_timeout_republish_time and not within_spec and not both_started:
                print("Republishing reset trajectories (possible action client issue). ")
                self._publish_reset_trajectories(new_base_reset_mat=False)
                pub_start_time = time.time()
            if cur_time - pub_start_time > reset_non_movement_republish_time and \
                (norm(init_dist_base_to_reset - dist_base_to_reset) < .01 and
                 norm(init_rot_dist_base_to_reset - rot_dist_base_to_reset) < .01 and
                 norm(init_dist_arm_to_init - dist_arm_to_init) < .01 and
                 norm(init_rot_dist_arm_to_init - rot_dist_arm_to_init) < .01) and not within_spec and not both_started:
                print("Reset trajectories not started, republishing (possible action client issue).")
                self._publish_reset_trajectories(new_base_reset_mat=False)
                pub_start_time = time.time()
            self.gui_lock.release()
            rospy.sleep(0.2)
            self.gui_lock.acquire()
            self.tf_odom_base.update()
            self.tf_base_tool.update()
            dist_arm_to_init, rot_dist_arm_to_init = rnt.pos_quat_np.get_trans_rot_dist(self.tf_base_tool.as_pos_quat(),
                                                 rnt.pos_quat_np.mat_to_pos_quat(self._reset_base_tool_mat))
            dist_base_to_reset, rot_dist_base_to_reset = rnt.pos_quat_np.get_trans_rot_dist(
                self.tf_odom_base.as_pos_quat(), rnt.pos_quat_np.mat_to_pos_quat(self.ep_odom_base_mat))
            # print('dists: ', dist_arm_to_init, rot_dist_arm_to_init, dist_base_to_reset, rot_dist_base_to_reset)
        print("Reset trajectories completed.")

        # ensure arm joint position is what it's supposed to be
        self.arm_joint_pos_lock.acquire()
        joint_pos_dist = norm(self._reset_joint_pos - self.arm_joint_pos_latest)
        self.arm_joint_pos_lock.release()
        if self.init_gripper_random_lim == (0, 0, 0, 0, 0, 0):
            assert joint_pos_dist < .1, "Arm EE pos is right, but joint positions are wrong. " \
                                        "Currently read joint positions are %s. Manually reset arm to " \
                                        "joint pos %s." % (self.arm_joint_pos_latest, self._reset_joint_pos)

        # TODO ensure gripper is in correct position

        # called after moving ee to init pose and user can now manually set up env objects
        if not self._reset_teleop_available:
            self._reset_helper()

        # if force torque is in state, reset it back to zero
        if 'force_torque' in self.state_data:
            self.pub_ft_zero.publish(True)
            self.gui_lock.release()
            time.sleep(0.5)  # zero reset node needs half a second of still robot to collect data to average
            self.gui_lock.acquire()

        # other resets
        self.ep_timesteps = 0
        self.prev_action = None
        self.prev_pose = None
        self.prev_grip_pos = None
        self.done_success = False
        self.done_failure = False
        self.done_timeout = False
        self.ft_lock.acquire()
        self.ft_filter = None
        self.ft_lock.release()

        # generate observation for return -- need published trajectories above to be completed
        obs, _ = self._prepare_obs()

        if self._reset_teleop_available:
            self.reset_teleop_complete = False

        self.gui_lock.release()
        self.pub_pos_limits.publish(self.pos_limits_marker)
        self._env_reset_complete = True

        return obs

    def _publish_reset_trajectories(self, new_base_reset_mat=True):
        self.tf_odom_tool.update()
        self.tf_odom_base.update()
        self.tf_base_tool.update()

        if self._moving_base:
            if new_base_reset_mat:
                self.ep_odom_base_mat = self._get_reset_base_mat()
                if self.env_to_gui_q is not None:
                    self.env_to_gui_q.put(dict(ep_odom_base_mat=self.ep_odom_base_mat))
            arm_path, base_path = self._get_combined_arm_base_reset_paths(self.ep_odom_base_mat)
        else:
            des_T = self.ep_odom_base_mat.dot(self._reset_base_tool_mat)
            arm_path = rnt.path.generate_smooth_path(
                first_frame=self.tf_odom_tool.transform,
                last_frame=rnt.tf_msg.mat_to_tf_msg(des_T),
                trans_velocity=self._reset_vel_trans, rot_velocity=self._reset_vel_rot,
                time_btwn_poses=self._time_between_poses_tc)

            base_path = rnt.thing.gen_no_movement_base_path(arm_path, self.ep_odom_base_mat)

        self.pub_base_traj.publish(base_path)
        self.pub_arm_traj.publish(arm_path)
        print("Arm & Base reset trajectory published.")

    def _prepare_obs(self):
        """ Order in returned state array: pose, prev_pose, grip_pos, prev_grip_pos, obj_pos, obj_rot"""
        return_obs = dict()
        return_arr = []

        # state data
        if self._poses_ref_frame == 'b':
            self.tf_base_tool.update()
            cur_pos_quat = self.tf_base_tool.as_pos_quat()
        elif self._poses_ref_frame == 'w':
            self.tf_odom_tool.update()
            cur_pos_quat = self.tf_odom_tool.as_pos_quat()

        # avoid quaternion values jumping
        if cur_pos_quat[-1] < 0:
            cur_pos_quat[3:] = -cur_pos_quat[3:]

        # fix pose to correspond to valid dofs
        cur_pose = cur_pos_quat[self.valid_act_t_dof.nonzero()]
        if sum(self.valid_act_r_dof) > 0:
            cur_pose = np.concatenate([cur_pose, cur_pos_quat[3:]])

        if 'pose' in self.state_data:
            return_obs['pose'] = cur_pose
            return_arr.append(cur_pose)

        if 'prev_pose' in self.state_data or self.position_impedance_control:
            if self.prev_pose is None:
                self.prev_pose = np.tile(cur_pose, (self.num_prev_pose + 1, 1))
            self.prev_pose = np.roll(self.prev_pose, 1, axis=0)
            self.prev_pose[0] = cur_pose
        if 'prev_pose' in self.state_data:
            return_obs['prev_pose'] = self.prev_pose[1:].flatten()
            return_arr.append(return_obs['prev_pose'])

        if 'grip_pos' in self.state_data or 'prev_grip_pos' in self.state_data:
            self.grip_lock.acquire()
            grip_pos = self.latest_grip
            self.grip_lock.release()
            if 'grip_pos' in self.state_data:
                return_obs['grip_pos'] = grip_pos
                return_arr.append(grip_pos)
            if 'prev_grip_pos' in self.state_data:
                if self.prev_grip_pos is None:
                    self.prev_grip_pos = np.tile(grip_pos, (self.num_prev_grip + 1, 1))
                self.prev_grip_pos = np.roll(self.prev_grip_pos, 1, axis=0)
                self.prev_grip_pos[0] = grip_pos
                return_obs['prev_grip_pos'] = np.array(self.prev_grip_pos[1:]).flatten()
                return_arr.append(return_obs['prev_grip_pos'])

        if 'force_torque' in self.state_data:
            self.ft_lock.acquire()
            return_obs['force_torque'] = self.latest_ft
            self.ft_lock.release()
            return_arr.append(return_obs['force_torque'])

        if 'timestep' in self.state_data:
            # adjust range of timesteps to be between -1 and 1
            adj_timestep = (self.ep_timesteps / self._max_episode_steps - .5) * 2
            return_obs['timestep'] = np.array([adj_timestep])
            return_arr.append(adj_timestep)

        if 'obj_pos' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')
        if 'obj_rot' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')
        if 'obj_rot_z' in self.state_data:
            raise NotImplementedError('Object positions not implemented, need to use ARtags or some other CV method.')

        return_arr = []
        for k in return_obs.keys():
            return_arr.append(return_obs[k])
        return_arr = np.concatenate(return_arr)

        # img and depth -- convert return structure to dict if they are included
        if self.img_in_state or self.depth_in_state:
            return_arr = dict(obs=return_arr)

        if self.img_in_state:
            self.img_lock.acquire()
        if self.depth_in_state:
            self.depth_lock.acquire()
        if self.img_in_state:
            return_obs['img'] = self.latest_img
            return_arr['img'] = self.latest_img
        if self.depth_in_state:
            return_obs['depth'] = self.latest_depth
            return_arr['depth'] = self.latest_depth
        if self.img_in_state:
            self.img_lock.release()
        if self.depth_in_state:
            self.depth_lock.release()

        return return_arr, return_obs

    def _get_reset_base_mat(self):
        raise NotImplementedError("Create a ThingRosMBEnv class to use a moving base.")

    def _get_combined_arm_base_reset_paths(self, reset_odom_base_mat):
        raise NotImplementedError("Create a ThingRosMBEnv class to use a moving base.")

    def set_reset_teleop_complete(self):
        self.reset_teleop_complete = True

    def set_done(self, success):
        """ Called by running script to manually set done to True based on either success or failure. Should
        be called before call to step. """
        screen_str = "successful" if success else "not successful"
        print("User manually says episode was " + screen_str)
        self.done_success = success
        self.done_failure = not success

    def _reset_helper(self):
        """ Called within reset, but to be overridden by child classes. This should somehow help the
        experimenter reset objects to a new random pose, possibly with instructions."""
        print('Warning: _reset_helper should be implemented by child classes.')

    def render(self, mode='human'):
        if mode == 'human':
            if self.gui_thread is None:  # future calls to render don't do anything
                self.gui_to_env_q = Queue()
                self.env_to_gui_q = Queue()
                self.gui_thread = Thread(target=self.gui_worker, args=(self.env_to_gui_q, self.gui_to_env_q))
                self.gui_thread.start()
                print('gui thread started')
                self.gui_get_data_thread = Thread(target=self._gui_data_worker)
                self.gui_get_data_thread.start()
                # self.gui_send_timer = rospy.Timer(rospy.Duration.from_sec(.1), self.__gui_timer_handler)

    def gui_worker(self, env_to_gui_q, gui_to_env_q):
        from PyQt5 import QtGui, QtCore, uic, QtWidgets
        from thing_gym_ros.envs.gui.generic import ThingRosGenericGui
        app = QtWidgets.QApplication(sys.argv)
        gui = ThingRosGenericGui(env_to_gui_q=env_to_gui_q, gui_to_env_q=gui_to_env_q, env_obj=self)
        gui.show()
        sys.exit(app.exec_())

    def _gui_data_worker(self):
        while(not rospy.is_shutdown()):
            gui_data_dict = self.gui_to_env_q.get()
            self.gui_lock.acquire()
            if gui_data_dict == 'close':  # means time to close
                print("Gui closed.")
                self.gui_lock.release()
                # self.gui_send_timer.shutdown()
                break
            for k in gui_data_dict:
                setattr(self, k, gui_data_dict[k])
            self.gui_lock.release()

    def __gui_timer_handler(self, e):
        # TODO delete this, may not be necessary for anything
        # put relevant data on queue -- have to be careful to make sure this doesn't slow down the env
        self.gui_lock.acquire()
        self.env_to_gui_q.put(dict(
            latest_processed_img=self.latest_processed_img
        ))
        self.gui_lock.release()

    def img_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg)

        if self._raw_img_width is None:
            self._raw_img_width = img.shape[1]
            self._raw_img_height = img.shape[0]

        if not self.sim and self._sensor == 'kinect':
            img_fixed = copy.deepcopy(img)
            img_fixed[:, :, 0] = img[:, :, 2]
            img_fixed[:, :, 2] = img[:, :, 0]
            img = img_fixed

        if self.image_center_crop != 1.0:
            img_cropped = thing_gym_ros_env_utils.center_crop_img(img, self.image_center_crop)
        elif self.image_crop != [[0, 0], [1, 1]]:
            img_cropped = thing_gym_ros_env_utils.crop_img(img, self.image_crop)
        else:
            img_cropped = img

        img_final = cv2.resize(img_cropped, (self.image_width, self.image_height))

        if self.env_to_gui_q is not None:
            self.env_to_gui_q.put(dict(env_img=img_final))

        self.img_lock.acquire()
        self.latest_img = img_final
        self.img_lock.release()

    def depth_cb(self, msg):
        depth = self.cv_bridge.imgmsg_to_cv2(msg)
        depth_fixed = copy.deepcopy(depth)

        if self._raw_depth_width is None:
            self._raw_depth_width = depth.shape[1]
            self._raw_depth_height = depth.shape[0]
        if self._img_depth_registered is None and self._require_img_depth_registration:
            if self._raw_img_width is not None and self._raw_depth_width is not None:
                assert self._raw_img_width == self._raw_depth_width and self._raw_img_height == self._raw_depth_height, \
                    "Image dimensions do not match depth dimensions, ensure that they are registered."
                self._img_depth_registered = True

        # want to fix all depth values to be float32 from 0 to 1, 0 = closest, 1 = max_dist
        if self.sim:
            max_depth = self._max_depth
        else:
            max_depth = 1000 * self._max_depth
        depth_fixed[np.isnan(depth)] = max_depth  # todo this should be interpolate instead
        depth_fixed[depth_fixed > max_depth] = max_depth
        depth_fixed = depth_fixed / max_depth

        if self.image_center_crop != 1.0:
            depth_cropped = thing_gym_ros_env_utils.center_crop_img(depth_fixed, self.image_center_crop)
        elif self.image_crop != [[0, 0], [1, 1]]:
            depth_cropped = thing_gym_ros_env_utils.crop_img(depth_fixed, self.image_crop)
        else:
            depth_cropped = depth_fixed

        depth_final = cv2.resize(depth_cropped, (self.image_width, self.image_height)).astype('float32')

        self.depth_lock.acquire()
        self.latest_depth = depth_final
        self.depth_lock.release()

    def grip_cb(self, msg):
        # msg is SModel_robot_input, or equivalently SModelRobotInput
        # each pos is a uint8, so normalize to range of [-1, 1] instead
        finger_a_pos = (msg.gPOA / 255 - .5) * 2
        finger_b_pos = (msg.gPOB / 255 - .5) * 2
        finger_c_pos = (msg.gPOC / 255 - .5) * 2
        self.grip_lock.acquire()
        self.latest_grip = np.array([finger_a_pos, finger_b_pos, finger_c_pos])
        self.latest_grip_bool = np.any(self.latest_grip > -.95)  # from real robot being fully open
        self.grip_lock.release()

    def ft_cb(self, msg):
        # to make data more friendly for learning, going to "normalize" it to be approximately within -1, 1
        # "a lot" of force is 50N, "a lot" of torque is 10, anything below 2.5N or .1Nm is considered noise, so cut off
        fo = msg.wrench.force
        to = msg.wrench.torque
        fo = np.array([fo.x, fo.y, fo.z])
        to = np.array([to.x, to.y, to.z])
        self.ft_lock.acquire()
        if self.ft_filter is None:
            self.ft_filter = np.tile(np.concatenate([fo, to]), [self.ft_filter_size, 1])
        else:
            self.ft_filter = np.roll(self.ft_filter, -1, axis=0)
            self.ft_filter[-1] = np.concatenate([fo, to])
        ft_filtered = np.median(self.ft_filter, axis=0)

        self.latest_ft_raw = ft_filtered
        self.ft_lock.release()

        # fix for env
        fo = ft_filtered[:3]
        to = ft_filtered[3:]
        fo[np.abs(fo) < 2.5] = 0
        to[np.abs(to) < .1] = 0
        force_fixed = fo / 50
        torque_fixed = to / 10
        self.ft_lock.acquire()
        self.latest_ft = np.concatenate([force_fixed, torque_fixed])
        self.ft_lock.release()

    def arm_joint_states_cb(self, msg):
        self.arm_joint_pos_lock.acquire()
        self.arm_joint_pos_latest = np.around(np.array(msg.actual.positions), decimals=3)
        self.arm_joint_pos_lock.release()

    def get_cur_base_tool_pose(self):
        self.tf_base_tool.update()
        return self.tf_base_tool.as_pos_quat()