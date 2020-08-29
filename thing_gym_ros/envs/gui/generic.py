# built-ins
import sys
import os
import queue

# ros
import rospy
import tf.transformations as tf_trans
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path

# other
from PyQt5 import QtWidgets, QtGui, QtCore, uic

# own
import ros_np_tools as rnt

# for type hint, otherwise creates circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv

path = os.path.dirname(os.path.abspath(__file__))
MainWindowUI, MainWindowBase = uic.loadUiType(os.path.join(path, 'thing_ros_generic.ui'))


class ThingRosGenericGui(MainWindowBase, MainWindowUI):
    def __init__(self, env_to_gui_q, gui_to_env_q, env_obj: 'ThingRosEnv'):
        super().__init__()
        self.setupUi(self)

        # play pause button
        self.play_pause_env.toggled.connect(self.handle_play_pause_env)
        self.playing = False

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.gui_update)
        self.update_timer.start(100)

        self.testing_group_box.hide()

        # ros publishers
        self.pub_servo = rospy.Publisher('/servo/command', Float64MultiArray, queue_size=1)
        self.pub_base_traj = rospy.Publisher('goal_ridgeback', Path, queue_size=1)
        self.pub_arm_traj = rospy.Publisher('goal_ur10', Path, queue_size=1)

        # queues
        self.env_to_gui_q = env_to_gui_q
        self.gui_to_env_q = gui_to_env_q

        # base buttons
        self.base_pose_before_adjustment = None
        self.set_base_adjustment_status(True)
        self.move_to_main_base_pose.clicked.connect(self.handle_move_to_main_base_pose)
        self.move_to_prev_base_pose.clicked.connect(self.handle_move_to_prev_base_pose)
        # self.move_to_main_arm_pose.clicked.connect(self.handle_move_to_main_arm_pose)
        self.main_base_to_current.clicked.connect(self.handle_main_base_to_current)
        # self.randomize_base_pose.clicked.connect(self.handle_randomize_base_pose)
        # self.test_theta_high.clicked.connect(self.handle_test_theta_high)
        # self.test_theta_low.clicked.connect(self.handle_test_theta_low)
        # self.base_trans_vel.setValue(self.env_obj._reset_base_vel_trans)
        # self.base_rot_vel.setValue(self.env_obj._reset_base_vel_rot)

    def gui_update(self):
        try:
            gui_data_dict = self.env_to_gui_q.get_nowait()
            odom_base_pos_quat = gui_data_dict['tf_odom_base_pos_quat']
            self.x_base_actual_line.setText("%3.4f" % odom_base_pos_quat[0])
            self.y_base_actual_line.setText("%3.4f" % odom_base_pos_quat[1])
            self.theta_base_actual_line.setText("%3.4f" % tf_trans.euler_from_quaternion(odom_base_pos_quat[3:])[2])

        except queue.Empty:
            pass

    def handle_play_pause_env(self, checked):
        if checked:
            self.play_pause_env.setStyleSheet("QPushButton { background-color : green; color : black; }")
            self.gui_to_env_q.put(dict(play_pause_env=True))
            self.set_base_adjustment_status(False)
        else:
            self.play_pause_env.setStyleSheet("QPushButton { background-color : white; color : black; }")
            self.gui_to_env_q.put(dict(play_pause_env=False))
            self.set_base_adjustment_status(True)

    def set_base_adjustment_status(self, allow):
        self.move_to_main_base_pose.setEnabled(allow)
        self.move_to_prev_base_pose.setEnabled(allow)
        self.x_base_main_spin_box.setReadOnly(not allow)
        self.y_base_main_spin_box.setReadOnly(not allow)
        self.theta_base_main_spin_box.setReadOnly(not allow)

    def handle_move_to_main_base_pose(self):
        """ Move the robot base to the current "main" base pose. """
        self.env_obj.tf_odom_base.update()
        self.env_obj.tf_odom_tool.update()

        base_path = rnt.path.generate_smooth_path(
            first_frame=self.env_obj.tf_odom_base.transform,
            last_frame=rnt.tf_msg.mat_to_tf_msg(self.env_obj._reset_odom_base_mat),
            trans_velocity=self.env_obj._reset_base_vel_trans,
            rot_velocity=self.env_obj._reset_base_vel_rot,
            time_btwn_poses=self.env_obj._time_between_poses_tc )

        # ensure that arm always maintains same pose relative to base
        arm_path = self.gen_no_movement_arm_path(base_path)
        #TODO adjust this if we change things so that we're not always running IK relative to odom

        self.statusBar().showMessage("Publishing robot trajectory to return base to main pose.")
        self.pub_arm_traj.publish(arm_path)
        self.pub_base_traj.publish(base_path)

    def handle_move_to_prev_base_pose(self):
        pass

    def handle_main_base_to_current(self):
        self.x_base_main_spin_box.setValue(float(self.x_base_actual_line.text()))
        self.y_base_main_spin_box.setValue(float(self.y_base_actual_line.text()))
        self.z_base_main_spin_box.setValue(float(self.z_base_actual_line.text()))

    def closeEvent(self, e):
        self.update_timer.stop()

        e.accept()
