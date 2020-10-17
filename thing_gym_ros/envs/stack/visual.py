from thing_gym_ros.envs.stack.generic import ThingRosStackGeneric
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


STACK_IMAGE_DEFAULTS = dict(
    img_in_state=True,
    depth_in_state=True,
    dense_reward=False,
    num_objs=2,
    state_data=('pose', 'grip_pos', 'prev_grip_pos', 'force_torque'),
    max_real_time=10,
    grip_in_action=True
)

STACK3_IMAGE_DEFAULTS = dict(**STACK_IMAGE_DEFAULTS)
STACK3_IMAGE_DEFAULTS['max_real_time'] = 20
STACK3_IMAGE_DEFAULTS['num_objs'] = 3

# ------------------------------------------------------------------------------------------------------------
# XYZ Image Envs
# ------------------------------------------------------------------------------------------------------------

STACK_XYZ_DEFAULTS = dict(
    valid_act_t_dof=(1, 1, 1),
    valid_act_r_dof=(0, 0, 0)
)

STACK_XYZ_IMAGE_DEFAULTS = dict(**STACK_IMAGE_DEFAULTS, **STACK_XYZ_DEFAULTS)
STACK3_XYZ_IMAGE_DEFAULTS = dict(**STACK3_IMAGE_DEFAULTS, **STACK_XYZ_DEFAULTS)


class ThingRosStack2XYZImage(ThingRosStackGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack2XYZImageMB(ThingRosStackGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack3XYZImage(ThingRosStackGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK3_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack3XYZImageMB(ThingRosStackGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK3_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

# ------------------------------------------------------------------------------------------------------------
# 6Dof Image Envs
# ------------------------------------------------------------------------------------------------------------

STACK_6DOF_DEFAULTS = dict(
    valid_act_t_dof=(1, 1, 1),
    valid_act_r_dof=(1, 1, 1)
)

STACK_6DOF_IMAGE_DEFAULTS = dict(**STACK_IMAGE_DEFAULTS, **STACK_6DOF_DEFAULTS)
STACK3_6DOF_IMAGE_DEFAULTS = dict(**STACK3_IMAGE_DEFAULTS, **STACK_6DOF_DEFAULTS)

class ThingRosStack26DOFImage(ThingRosStackGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack26DOFImageMB(ThingRosStackGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack36DOFImage(ThingRosStackGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK3_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

class ThingRosStack36DOFImageMB(ThingRosStackGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False):

        super().__init__(**STACK3_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)