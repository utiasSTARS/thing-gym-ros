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
from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv


class ThingRosMBEnv(ThingRosEnv):
    def __init__(self, *args, **kwargs):
        """ A generic thing env that allows for moving the base between episodes. """
        super().__init__(*args, **kwargs)

        # extra tf objs -- need (reasonably) calibrated camera
        self.tf_odom_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'camera_link')
        self.tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')

        # extra publishers
        self.pub_base_traj = rospy.Publisher('goal_ridgeback', Path, queue_size=1)

        self._main_odom_base_mat = self.tf_odom_base.as_mat()
        self._main_odom_base_pos_eul = self.tf_odom_base.as_pos_eul()
        self._base_theta_maximums = self.cfg['base_theta_maximums']  # each in rads
        self._cam_workspace_distance = self.cfg['cam_workspace_distance']
        self._base_reset_noise = self.cfg['base_reset_noise']

        # publish a tf that shows where the currently estimated workspace center is so that
        # we can ensure we reset the base pose properly
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')
        self.workspace_center_tf_msg = TransformStamped()
        self.workspace_center_tf_msg.header.frame_id = 'odom'
        self.workspace_center_tf_msg.child_frame_id = 'workspace_center'
        T_update = np.eye(4)
        if self.sim:
            T_update[0, 3] = self._cam_workspace_distance
        else:
            T_update[2, 3] = self._cam_workspace_distance
        T_workspace_center = self.tf_odom_cam.as_mat().dot(T_update)
        self.workspace_center_tf_msg.transform = rnt.tf_msg.mat_to_tf_msg(T_workspace_center)
        self.workspace_center_tf_msg.header.stamp = rospy.Time.now()
        self.tf_static_broadcaster.sendTransform(self.workspace_center_tf_msg)

        self._moving_base = True

    def _get_reset_base_mat(self):

        self.tf_odom_base.update()
        odom_base_pos_eul = self.tf_odom_base.as_pos_eul()
        main_theta = self._main_odom_base_pos_eul[5]
        theta = self.np_random.uniform(
            low=main_theta + self._base_theta_maximums[0],
            high=main_theta + self._base_theta_maximums[1],
            size=1
        )
        b_x, b_y = self.gen_base_tf_from_theta_and_ws_center(theta)
        noise = self._base_reset_noise
        (b_x, b_y) = np.array([b_x, b_y]).squeeze() + self.np_random.uniform(low=-noise, high=noise, size=2)
        rb_z = theta + self.np_random.uniform(low=-noise, high=noise, size=1)
        pos_eul = [b_x, b_y, 0, 0, 0, rb_z]
        return rnt.tf_mat.pos_eul_to_mat(pos_eul)


    def _publish_reset_base_path(self, reset_base_mat):
        des_base_tf_msg = rnt.tf_msg.mat_to_tf_msg(reset_base_mat)
        base_path = rnt.path.generate_smooth_path(self.tf_odom_base.transform, des_base_tf_msg,
                                                  self._reset_base_vel_trans, self._reset_base_vel_rot,
                                                  self._time_between_poses_tc)

        self.pub_base_traj.publish(base_path)

    def gen_base_tf_from_theta_and_ws_center(self, theta, cam_forward_axis='z'):
        """ Generate a base pose from a chosen value for theta while maintaining the
        workspace center constraint (so the camera still points at the workspace center). """

        self.tf_base_cam.update()

        T_bc = self.tf_base_cam.as_mat()
        u_d = self._cam_workspace_distance
        u_x, u_y, u_z = self.workspace_center_tf_msg.transform.translation.x, \
                        self.workspace_center_tf_msg.transform.translation.y, \
                        self.workspace_center_tf_msg.transform.translation.z

        if cam_forward_axis == 'x':
            b_z = -u_d * T_bc[2, 0] - T_bc[2, 3] + u_z
            b_x = np.sin(theta) * (u_d * T_bc[1, 0] + T_bc[1, 3]) - \
                  np.cos(theta) * (u_d * T_bc[0, 0] + T_bc[0, 3]) + u_x
            b_y = -np.cos(theta) * (u_d * T_bc[1, 0] + T_bc[1, 3]) - \
                  np.sin(theta) * (u_d * T_bc[0, 0] + T_bc[0, 3]) + u_y
        elif cam_forward_axis == 'y':
            b_z = -u_d * T_bc[2, 1] - T_bc[2, 3] + u_z
            b_x = np.sin(theta) * (u_d * T_bc[1, 1] + T_bc[1, 3]) - \
                  np.cos(theta) * (u_d * T_bc[0, 1] + T_bc[0, 3]) + u_x
            b_y = -np.cos(theta) * (u_d * T_bc[1, 1] + T_bc[1, 3]) - \
                  np.sin(theta) * (u_d * T_bc[0, 1] + T_bc[0, 3]) + u_y
        elif cam_forward_axis == 'z':
            b_z = -u_d * T_bc[2, 2] - T_bc[2, 3] + u_z
            b_x = np.sin(theta) * (u_d * T_bc[1, 2] + T_bc[1, 3]) - \
                  np.cos(theta) * (u_d * T_bc[0, 2] + T_bc[0, 3]) + u_x
            b_y = -np.cos(theta) * (u_d * T_bc[1, 2] + T_bc[1, 3]) - \
                  np.sin(theta) * (u_d * T_bc[0, 2] + T_bc[0, 3]) + u_y
        else:
            raise NotImplementedError("cam_forward axis must be string of x, y, or z, not %s"
                                      % str(cam_forward_axis))

        return b_x, b_y