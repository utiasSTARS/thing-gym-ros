# built-ins
import sys
import os
import copy
import time
# from multiprocessing import Lock
from threading import Thread, Lock
import queue
from queue import Queue

# ros
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Transform, TransformStamped
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Path
import tf2_ros
import tf.transformations as tf_trans
import rostopic
from std_msgs.msg import Float64MultiArray, Bool

# other
import numpy as np
from numpy.linalg import norm
import gym
from gym.utils import seeding
from gym import spaces
import pygame
import yaml

# local
import ros_np_tools as rnt
from thing_gym_ros_catkin.msg import KeyboardTrigger


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
                 state_data=('pose', 'prev_pose', 'grip_pos', 'obj_pos', 'obj_rot'),
                 valid_act_t_dof=(1, 1, 1),
                 valid_act_r_dof=(1, 1, 1),
                 num_prev_pos=5,
                 gap_between_prev_pos=.1,  # in seconds
                 # moving_base=False,  # whether base moves to different views between episodes
                 max_real_time=5,  # in seconds
                 success_causes_done=False,
                 failure_causes_done=False,
                 reset_teleop_available=False
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
                                                          (self.cfg['control_type'], ThingRosEnv.CONTROL_TYPES)
        self._control_type = self.cfg['control_type']
        self._poses_ref_frame = self.cfg['poses_ref_frame']
        self._rot_act_rep = self.cfg['rot_act_rep']
        self.grip_in_action = grip_in_action
        self.default_grip_state = default_grip_state
        self.state_data = state_data
        self.img_in_state = img_in_state
        self.depth_in_state = depth_in_state
        self.image_width = self.cfg['img_width']
        self.image_height = self.cfg['img_height']
        self.image_zoom_crop = self.cfg['img_zoom_crop']
        self._control_freq = self.cfg['control_freq']
        self._max_episode_steps = max_real_time * self._control_freq
        self.valid_act_t_dof = valid_act_t_dof
        self.valid_act_r_dof = valid_act_r_dof
        self.pos_limits = self.cfg['pos_limits']
        self.arm_max_trans_vel = self.cfg['arm_max_trans_vel']
        self.arm_max_rot_vel = self.cfg['arm_max_rot_vel']
        self._moving_base = False
        if self._rot_act_rep == 'quat':
            self._quat_in_action = True
            raise NotImplementedError('Implement if needed')
        else:
            self._quat_in_action = False

        # varibles
        self._timestep = 0

        # gym setup
        self._num_trans = sum(self.valid_act_t_dof)
        if sum(self.valid_act_r_dof) > 0 and self._quat_in_action:
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

        # hard-coded parameters from thing_control -- time_from_start in servoCallback
        # should be set to the same value as control freq, ideally, time_between_poses is used for full paths
        # for e.g. resetting
        self._action_duration = .1
        self._time_between_poses_tc = .3

        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # tf objs for updating poses
        self.tf_odom_base = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'base_link')
        self.tf_base_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'thing_tool')
        self.tf_odom_tool = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'thing_tool')

        # subscribers
        if img_in_state:  # defaults are based on using a kinect
            if self.cfg['alt_img_topic'] is None:
                img_topic = '/camera/rgb/image_raw' if self.sim else '/camera/sd/image_color_rect'
            else:
                img_topic = self.cfg['alt_img_topic']
            self.sub_img = rospy.Subscriber(img_topic, Image, self.img_cb)
            self.img_lock = Lock()
            self.latest_img = None
        if depth_in_state:
            if self.cfg['alt_depth_topic'] is None:
                depth_topic = '/camera/depth/image_raw' if self.sim else '/camera/sd/image_depth_rect'
            else:
                depth_topic = self.cfg['depth_topic']
            self.sub_depth = rospy.Subscriber(depth_topic, Image, self.depth_cb)
            self.depth_lock = Lock()
            self.latest_depth = None

        # publishers
        self.pub_servo = rospy.Publisher('/servo/command', Float64MultiArray, queue_size=1)
        self.pub_gripper = rospy.Publisher('FRL/remote_trigger', KeyboardTrigger, queue_size=10)
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
        self._max_reset_trans = 1.0  # meters
        self._max_reset_rot = 2.0  # radians
        self._reset_vel_trans = .15  # m/s
        self._reset_vel_rot = .5  # rad/s
        self._reset_base_vel_trans = .15  # m/s
        self._reset_base_vel_rot = .3  # rad/s
        self._reset_teleop_available = reset_teleop_available
        if reset_teleop_available:
            self.reset_teleop_complete = False

        # base movement params
        # if self._moving_base:
        #     self.tf_odom_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'camera_link')
        #     self.tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')
        #     self._main_odom_base_mat = self.tf_odom_base.as_mat()
        #     self._main_odom_base_pos_eul = self.tf_odom_base.as_pos_eul()
        #     self._base_theta_maximums = self.cfg['base_theta_maximums']  # each in rads
        #     self._cam_workspace_distance = self.cfg['cam_workspace_distance']
        #     self._base_reset_noise = self.cfg['base_reset_noise']
        #
        #     # publish a tf that shows where the currently estimated workspace center is so that
        #     # we can ensure we reset the base pose properly
        #     self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        #     self.tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')
        #     self.workspace_center_tf_msg = TransformStamped()
        #     self.workspace_center_tf_msg.header.frame_id = 'odom'
        #     self.workspace_center_tf_msg.child_frame_id = 'workspace_center'
        #     T_update = np.eye(4)
        #     if self.sim:
        #         T_update[0, 3] = self._cam_workspace_distance
        #     else:
        #         T_update[2, 3] = self._cam_workspace_distance
        #     T_workspace_center = self.tf_odom_cam.as_mat().dot(T_update)
        #     self.workspace_center_tf_msg.transform = rnt.tf_msg.mat_to_tf_msg(T_workspace_center)
        #     self.workspace_center_tf_msg.header.stamp = rospy.Time.now()
        #     self.tf_static_broadcaster.sendTransform(self.workspace_center_tf_msg)

        # gui
        self.gui_thread = None
        self.env_to_gui_q = None
        self.gui_to_env_q = None
        self.gui_timer = None
        self.gui_dict = None
        self.gui_lock = Lock()
        self.play_pause_env = True
        self.latest_processed_img = None

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

        # gui handling
        if self.gui_thread is not None:
            self.gui_lock.acquire()
            if not self.play_pause_env:
                self.gui_lock.release()
                print("Env is paused, unpause using gui.")
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
            T_delta_rot = tf_trans.rotation_matrix(ang, axis)

            # we want movement to happen starting from the current tool position, but using the axes of the
            # current ref frame
            T_odom_tool = self.tf_odom_tool.as_mat()

            T_act_frame = copy.deepcopy(T_odom_tool)
            if self._poses_ref_frame == 'b':
                T_base_tool = self.tf_base_tool.as_mat()
                T_act_frame[:3, :3] = T_base_tool[:3, :3]
                R_ref_tool = self.tf_base_tool.as_mat()[:3, :3]
            elif self._poses_ref_frame == 'w':
                T_act_frame[:3, :3] = np.eye(3)
                R_ref_tool = T_odom_tool[:3, :3]
            T_new = T_act_frame.dot(T_delta_trans)
            T_new[:3, :3] = T_delta_rot.dot(R_ref_tool)

            # T_new, limit_reached = self._limit_action(T_new)  # TODO implement this

            servo_msg = rnt.thing.get_servo_msg(mat=T_new, base_tf_msg=self.tf_odom_base.transform)

        # process grip action
        if self.grip_in_action:
            if self.default_grip_state == 'o':
                grip = 'c' if action[-1] > 0 else 'o'
            else:
                grip = 'o' if action[-1] > 0 else 'c'
        else:
            grip = self.default_grip_state
        g_msg = KeyboardTrigger()
        g_msg.label = grip

        import ipdb; ipdb.set_trace()

        self.pub_servo.publish(servo_msg)
        self.pub_gripper.publish(g_msg)

        self.gui_lock.release()
        rospy.sleep(self._fixed_time_after_action)
        self.gui_lock.acquire()

        # get and process observation
        obs = None

        self.gui_lock.release()
        self.rate.sleep()
        self.gui_lock.acquire()


        # get reward
        r = None

        # get done
        self._timestep += 1
        done = self._timestep == self._max_episode_steps

        info = None

        return obs, r, done, info

    def _limit_action(self, action):
        """ Limit the desired action based on pos and vel maximums. """
        #TODO

    def reset(self):
        """ Reset the environment to the beginning of an episode.

        In sim, a user could theoretically reload or otherwise move objects arbitrarily, but since the
        primary use for this class is for the real robot, this method will require interaction with a person."""

        self.gui_lock.acquire()

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

        # first do safety checks to make sure movement isn't too dramatic
        self.tf_base_tool.update()
        base_tool_mat = self.tf_base_tool.as_mat()
        base_tool_pos_quat = self.tf_base_tool.as_pos_quat()
        dist_arm_to_init = norm(base_tool_pos_quat[:3] - self._reset_base_tool_mat[3, :3])
        if dist_arm_to_init > self._max_reset_trans:
            raise RuntimeError("EE is %.3fm from initial pose. Must be within %.3fm to reset." %
                               (dist_arm_to_init, self._max_reset_trans))
            # input("Move arm closer to initial pose and press Enter to continue...")
            # self.tf_base_tool.update()
            # dist_arm_to_init = norm(base_tool_pos_quat[:3] - self._reset_base_tool_mat[3, :3])
        rot_dist_arm_to_init = np.arccos(
            np.clip((np.trace(np.dot(base_tool_mat[:3, :3], self._reset_base_tool_mat[:3, :3].T)) - 1) / 2, -1.0, 1.0))
        if rot_dist_arm_to_init > self._max_reset_rot:
            raise RuntimeError("EE is %.3frad from init pose. Must be within %.3frad." %
                               (rot_dist_arm_to_init, self._max_reset_rot))
            # input("Move arm closer to initial pose and press Enter to continue...")
            # self.tf_base_tool.update()
            # rot_dist_arm_to_init = np.arccos(
            #     np.clip((np.trace(np.dot(base_tool_mat[:3, :3], self._reset_base_tool_mat[:3, :3].T)) - 1) / 2, -1.0,
            #             1.0))

        # complete the movement -- thing_control takes all motions in the frame of odom
        reset_thresh = .01
        self.gui_lock.release()
        input("Ensure there is a linear, collision-free path between end-effector and initial pose,"
              "then press Enter to continue...")
        self.gui_lock.acquire()
        print("Publishing arm reset movement.")
        self.tf_odom_tool.update()
        self.tf_odom_base.update()

        if self._moving_base:
            T_odom_base = self._get_reset_base_mat()
            self._publish_reset_base_path(T_odom_base)
        else:
            T_odom_base = self.tf_odom_base.as_mat()
        T_des_odom_tool = T_odom_base.dot(self._reset_base_tool_mat)
        arm_path = rnt.path.generate_smooth_path(
            first_frame=self.tf_odom_tool.transform,
            last_frame=rnt.tf_msg.mat_to_tf_msg(T_des_odom_tool),
            trans_velocity=self._reset_vel_trans,
            rot_velocity=self._reset_vel_rot,
            time_btwn_poses=self._time_between_poses_tc)

        self.pub_arm_traj.publish(arm_path)
        print("Reset trajectory published.")

        # called after moving ee to init pose and user can now manually set up env objects
        if not self._reset_teleop_available:
            self._reset_helper()

        # generate observation for return -- need published trajectories above to be completed
        # todo continue here
        obs = None

        # other resets
        self._timestep = 0

        self.gui_lock.release()

        return obs

    def _get_reset_base_mat(self):
        raise NotImplementedError("Create a ThingRosMBEnv class to use a moving base.")

    def _publish_reset_base_path(self, reset_base_mat):
        raise NotImplementedError("Create a ThingRosMBEnv class to use a moving base.")

    def set_reset_teleop_complete(self):
        self.reset_teleop_complete = True

    def _reset_helper(self):
        """ Called within reset, but to be overridden by child classes. This should somehow help the
        experimenter reset objects to a new random pose, possibly with instructions."""
        raise NotImplementedError('_reset_helper should be implemented by child classes.')

    def render(self, mode='human'):
        if mode == 'human':
            if self.gui_thread is None:
                self.gui_to_env_q = Queue()
                self.env_to_gui_q = Queue()
                self.gui_thread = Thread(target=self.gui_worker, args=(self.env_to_gui_q, self.gui_to_env_q))
                self.gui_thread.start()
                print('gui thread started')
                self.gui_get_data_thread = Thread(target=self._gui_data_worker)
                self.gui_get_data_thread.start()
                self.gui_send_timer = rospy.Timer(rospy.Duration.from_sec(.1), self.__gui_timer_handler)

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
            if gui_data_dict == None:  # means time to close
                break
            for k in gui_data_dict:
                setattr(self, k, gui_data_dict[k])
            self.gui_lock.release()

    def __gui_timer_handler(self, e):

        # put relevant data on queue -- have to be careful to make sure this doesn't slow down the env
        self.gui_lock.acquire()
        self.env_to_gui_q.put(dict(
            latest_processed_img=self.latest_processed_img
        ))
        self.gui_lock.release()

    def img_cb(self, msg):
        # todo check image processing time, if it's fast enough do it in here
        # if self.env_to_gui_q is not None:
        #     self.env_to_gui_q.put(dict(env_img=img))
        pass

    def depth_cb(self, msg):
        pass