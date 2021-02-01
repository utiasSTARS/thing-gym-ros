import os

from thing_gym_ros.envs.thing_ros_generic import XYZ_DEFAULTS, SIXDOF_DEFAULTS
from thing_gym_ros.envs.door.generic import ThingRosDoorGeneric
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


DOOR_IMAGE_DEFAULTS = dict(
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

DOOR_XYZ_IMAGE_DEFAULTS = dict(**DOOR_IMAGE_DEFAULTS, **XYZ_DEFAULTS)

class ThingRosDoorXYZImage(ThingRosDoorGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DOOR_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDoorXYZImageMB(ThingRosDoorGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DOOR_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

# ------------------------------------------------------------------------------------------------------------
# 6Dof Image Envs
# ------------------------------------------------------------------------------------------------------------

DOOR_6DOF_IMAGE_DEFAULTS = dict(**DOOR_IMAGE_DEFAULTS, **SIXDOF_DEFAULTS)

class ThingRosDoor6DOFImage(ThingRosDoorGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DOOR_6DOF_IMAGE_DEFAULTS,
                         success_causes_done=True,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDoor6DOFImageMB(ThingRosDoorGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**DOOR_6DOF_IMAGE_DEFAULTS,
                         success_causes_done=True,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosDoorRanGrip6DOFImage(ThingRosDoorGeneric):
  def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
    super().__init__(**DOOR_6DOF_IMAGE_DEFAULTS,
                     success_causes_done=True,
                     reset_teleop_available=reset_teleop_available,
                     success_feedback_available=success_feedback_available,
                     init_gripper_random_lim=(.12, .05, .05, 0, 0, 0), **kwargs)

class ThingRosDoorRanGrip6DOFImageMB(ThingRosDoorGeneric, ThingRosMBEnv):
  def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
    super().__init__(**DOOR_6DOF_IMAGE_DEFAULTS,
                     success_causes_done=True,
                     reset_teleop_available=reset_teleop_available,
                     success_feedback_available=success_feedback_available,
                     init_gripper_random_lim=(.12, .05, .05, 0, 0, 0), **kwargs)