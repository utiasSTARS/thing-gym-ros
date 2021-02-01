import os

from thing_gym_ros.envs.thing_ros_generic import XYZ_DEFAULTS, SIXDOF_DEFAULTS
from thing_gym_ros.envs.drawer.generic import ThingRosDrawerGeneric
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


DRAWER_IMAGE_DEFAULTS = dict(
    img_in_state=True,
    depth_in_state=True,
    dense_reward=False,
    num_objs=1,
    # state_data=('pose', 'grip_pos', 'prev_grip_pos', 'force_torque', 'timestep'),
    state_data=('pose', 'force_torque'),
    max_real_time=15,
    grip_in_action=False,
    default_grip_state='o',

)

# ------------------------------------------------------------------------------------------------------------
# XYZ Image Envs
# ------------------------------------------------------------------------------------------------------------

DRAWER_XYZ_IMAGE_DEFAULTS = dict(**DRAWER_IMAGE_DEFAULTS, **XYZ_DEFAULTS)

class ThingRosDrawerXYZImage(ThingRosDrawerGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DRAWER_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDrawerXYZImageMB(ThingRosDrawerGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DRAWER_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

# ------------------------------------------------------------------------------------------------------------
# 6Dof Image Envs
# ------------------------------------------------------------------------------------------------------------

DRAWER_6DOF_IMAGE_DEFAULTS = dict(**DRAWER_IMAGE_DEFAULTS, **SIXDOF_DEFAULTS)

class ThingRosDrawer6DOFImage(ThingRosDrawerGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DRAWER_6DOF_IMAGE_DEFAULTS,
                         success_causes_done=True,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDrawer6DOFImageMB(ThingRosDrawerGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DRAWER_6DOF_IMAGE_DEFAULTS,
                         success_causes_done=True,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDrawerRanGrip6DOFImage(ThingRosDrawerGeneric):
  def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
    super().__init__(**DRAWER_6DOF_IMAGE_DEFAULTS,
                     success_causes_done=True,
                     reset_teleop_available=reset_teleop_available,
                     success_feedback_available=success_feedback_available,
                     init_gripper_random_lim=(.12, .05, .05, 0, 0, 0), **kwargs)

class ThingRosDrawerRanGrip6DOFImageMB(ThingRosDrawerGeneric, ThingRosMBEnv):
  def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
    super().__init__(**DRAWER_6DOF_IMAGE_DEFAULTS,
                     success_causes_done=True,
                     reset_teleop_available=reset_teleop_available,
                     success_feedback_available=success_feedback_available,
                     init_gripper_random_lim=(.12, .05, .05, 0, 0, 0), **kwargs)