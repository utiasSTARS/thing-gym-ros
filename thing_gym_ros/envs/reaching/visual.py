from thing_gym_ros.envs.reaching.generic import ThingRosReachingGeneric
from thing_gym_ros.envs.thing_ros_mb_generic import ThingRosMBEnv


REACHING_IMAGE_DEFAULTS = dict(
    img_in_state=True,
    depth_in_state=True,
    dense_reward=False,
    num_objs=1,
    state_data=('pose'),
    max_real_time=5,
    grip_in_action=False
)

REACH_AND_GRASP_IMAGE_DEFAULTS = dict(**REACHING_IMAGE_DEFAULTS)
REACH_AND_GRASP_IMAGE_DEFAULTS['grip_in_action'] = True
REACH_AND_GRASP_IMAGE_DEFAULTS['state_data'] = ('pose', 'grip_pos', 'prev_grip_pos')

# ------------------------------------------------------------------------------------------------------------
# XY Image Envs
# ------------------------------------------------------------------------------------------------------------

REACHING_XY_DEFAULTS = dict(
    valid_act_t_dof=(1, 1, 0),
    valid_act_r_dof=(0, 0, 0)
)

REACHING_XY_IMAGE_DEFAULTS = dict(**REACHING_IMAGE_DEFAULTS, **REACHING_XY_DEFAULTS)
REACH_AND_GRASP_XY_IMAGE_DEFAULTS = dict(**REACH_AND_GRASP_IMAGE_DEFAULTS, **REACHING_XY_DEFAULTS)

class ThingRosReachingXYImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_XY_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachingXYImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_XY_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGraspXYImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_XY_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGraspXYImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_XY_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

# ------------------------------------------------------------------------------------------------------------
# XYZ Image Envs
# ------------------------------------------------------------------------------------------------------------

REACHING_XYZ_DEFAULTS = dict(
    valid_act_t_dof=(1, 1, 1),
    valid_act_r_dof=(0, 0, 0)
)

REACHING_XYZ_IMAGE_DEFAULTS = dict(**REACHING_IMAGE_DEFAULTS, **REACHING_XYZ_DEFAULTS)
REACH_AND_GRASP_XYZ_IMAGE_DEFAULTS = dict(**REACH_AND_GRASP_IMAGE_DEFAULTS, **REACHING_XYZ_DEFAULTS)

class ThingRosReachingXYZImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachingXYZImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGraspXYZImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGraspXYZImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_XYZ_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

# ------------------------------------------------------------------------------------------------------------
# 6Dof Image Envs
# ------------------------------------------------------------------------------------------------------------

REACHING_6DOF_DEFAULTS = dict(
    valid_act_t_dof=(1, 1, 1),
    valid_act_r_dof=(1, 1, 1)
)

REACHING_6DOF_IMAGE_DEFAULTS = dict(**REACHING_IMAGE_DEFAULTS, **REACHING_6DOF_DEFAULTS)
REACH_AND_GRASP_6DOF_IMAGE_DEFAULTS = dict(**REACH_AND_GRASP_IMAGE_DEFAULTS, **REACHING_6DOF_DEFAULTS)

class ThingRosReaching6DOFImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReaching6DOFImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACHING_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGrasp6DOFImage(ThingRosReachingGeneric):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)

class ThingRosReachAndGrasp6DOFImageMB(ThingRosReachingGeneric, ThingRosMBEnv):
    def __init__(self, reset_teleop_available=False, success_feedback_available=False, **kwargs):

        super().__init__(**REACH_AND_GRASP_6DOF_IMAGE_DEFAULTS,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available, **kwargs)