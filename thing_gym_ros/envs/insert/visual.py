import os

from thing_gym_ros.envs.thing_ros_generic import XYZ_DEFAULTS, SIXDOF_DEFAULTS
from thing_gym_ros.envs.insert.generic import ThingRosInsertGeneric
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


INSERT_IMAGE_DEFAULTS = dict(
    img_in_state=True,
    depth_in_state=True,
    dense_reward=False,
    num_objs=1,
    # state_data=('pose', 'grip_pos', 'prev_grip_pos', 'force_torque', 'timestep'),
    state_data=('pose', 'grip_pos', 'prev_grip_pos', 'force_torque'),
    max_real_time=10,
    grip_in_action=False,
    default_grip_state='c',

)

PICK_AND_INSERT_IMAGE_DEFAULTS = dict(**INSERT_IMAGE_DEFAULTS)
PICK_AND_INSERT_IMAGE_DEFAULTS['grip_in_action'] = True
PICK_AND_INSERT_IMAGE_DEFAULTS['max_real_time'] = 15
PICK_AND_INSERT_IMAGE_DEFAULTS['default_grip_state'] = 'o'


# ------------------------------------------------------------------------------------------------------------
# XYZ Image Envs
# ------------------------------------------------------------------------------------------------------------

INSERT_XYZ_IMAGE_DEFAULTS = dict(**INSERT_IMAGE_DEFAULTS, **XYZ_DEFAULTS)
PICK_AND_INSERT_XYZ_IMAGE_DEFAULTS = dict(**PICK_AND_INSERT_IMAGE_DEFAULTS, **XYZ_DEFAULTS)


class ThingRosInsertXYZImage(ThingRosInsertGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosInsertXYZImageMB(ThingRosInsertGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosPickAndInsertXYZImage(ThingRosInsertGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosPickAndInsertXYZImageMB(ThingRosInsertGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

# ------------------------------------------------------------------------------------------------------------
# 6Dof Image Envs
# ------------------------------------------------------------------------------------------------------------


INSERT_6DOF_IMAGE_DEFAULTS = dict(**INSERT_IMAGE_DEFAULTS, **SIXDOF_DEFAULTS)
PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS = dict(**PICK_AND_INSERT_IMAGE_DEFAULTS, **SIXDOF_DEFAULTS)

class ThingRosInsert6DOFImage(ThingRosInsertGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosInsert6DOFImageMB(ThingRosInsertGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**INSERT_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosPickAndInsert6DOFImage(ThingRosInsertGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosPickAndInsert6DOFImageMB(ThingRosInsertGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
        super().__init__(**PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)


class ThingRosPickAndInsertCloser6DOFImageMB(ThingRosInsertGeneric, ThingRosMBEnv):
  def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):
    robot_config_file = os.path.join(os.path.dirname(__file__), "insert_new_config.yaml")
    # PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS['state_data'] = ('pose', 'grip_pos', 'prev_grip_pos', 'force_torque', 'timestep')
    PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS['state_data'] = ('pose', 'grip_pos', 'prev_grip_pos', 'force_torque')
    super().__init__(**PICK_AND_INSERT_6DOF_IMAGE_DEFAULTS,
                     success_causes_done=True,
                     reset_teleop_available=reset_teleop_available,
                     success_feedback_available=success_feedback_available,
                     robot_config_file=robot_config_file, **kwargs)