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

# local
import ros_np_tools as rnt
from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv


class ThingRosMBEnv(ThingRosEnv):
    def __init__(self, *args, **kwargs):
        """ A generic thing env that allows for moving the base between episodes. """
        super().__init__(*args, **kwargs)

        if kwargs['info_env_only']:
            return

        # extra tf objs -- need (reasonably) calibrated camera
        self.tf_odom_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'odom', 'camera_link')
        self.tf_base_cam = rnt.tf_msg.TransformWithUpdate(self.tf_buffer, 'base_link', 'camera_link')

        # traj publishers for resetting
        self.pub_base_traj = rospy.Publisher('goal_ridgeback', Path, queue_size=1)
        self.pub_arm_traj = rospy.Publisher('goal_ur10', Path, queue_size=1)

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
        update_axis_dict = dict(x=0, y=1, z=2)
        T_update[update_axis_dict[self._cam_forward_axis], 3] = self._cam_workspace_distance
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
        des_theta = main_theta + theta
        b_x, b_y = self.gen_base_tf_from_theta_and_ws_center(des_theta, cam_forward_axis=self._cam_forward_axis)
        noise = self._base_reset_noise
        (b_x, b_y) = np.array([b_x, b_y]).squeeze() + self.np_random.uniform(low=-noise, high=noise, size=2)
        rb_z = des_theta + self.np_random.uniform(low=-noise, high=noise, size=1)
        pos_eul = [b_x, b_y, 0, 0, 0, rb_z]
        return rnt.tf_mat.pos_eul_to_mat(pos_eul)

    def _get_combined_arm_base_reset_paths(self, reset_odom_base_mat):
        reset_base_tool_tf_msg = rnt.tf_msg.mat_to_tf_msg(self._reset_base_tool_mat)
        reset_odom_base_tf_msg = rnt.tf_msg.mat_to_tf_msg(reset_odom_base_mat)

        # get number of traj points for arm and for base based on desired velocities, then
        # generate smooth paths based on whichever has more points (so no velocity maximums are exceeded)
        num_pts_arm = rnt.path.get_num_path_pts(
            first_frame=self.tf_base_tool.transform, last_frame=reset_base_tool_tf_msg,
            trans_velocity=self._reset_vel_trans, rot_velocity=self._reset_vel_rot,
            time_btwn_poses=self._time_between_poses_tc)
        num_pts_base = rnt.path.get_num_path_pts(
            first_frame=self.tf_odom_base.transform, last_frame=reset_odom_base_tf_msg,
            trans_velocity=self._reset_base_vel_trans, rot_velocity=self._reset_base_vel_rot,
            time_btwn_poses=self._time_between_poses_tc)
        if num_pts_arm > num_pts_base:
            arm_path_base_ref = rnt.path.generate_smooth_path(
                first_frame=self.tf_base_tool.transform, last_frame=reset_base_tool_tf_msg,
                trans_velocity=self._reset_vel_trans, rot_velocity=self._reset_vel_rot,
                time_btwn_poses=self._time_between_poses_tc)
            base_path = rnt.path.generate_smooth_path(
                first_frame=self.tf_odom_base.transform, last_frame=reset_odom_base_tf_msg, num_pts=num_pts_arm)
        else:
            arm_path_base_ref = rnt.path.generate_smooth_path(
                first_frame=self.tf_base_tool.transform, last_frame=reset_base_tool_tf_msg, num_pts=num_pts_base)
            base_path = rnt.path.generate_smooth_path(
                first_frame=self.tf_odom_base.transform, last_frame=reset_odom_base_tf_msg,
                trans_velocity=self._reset_base_vel_trans, rot_velocity=self._reset_base_vel_rot,
                time_btwn_poses=self._time_between_poses_tc)
        arm_path = rnt.path.get_new_bases_for_path(arm_path_base_ref, base_path)
        return arm_path, base_path

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