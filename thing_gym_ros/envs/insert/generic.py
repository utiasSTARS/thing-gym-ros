import numpy as np

from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv


class ThingRosInsertGeneric(ThingRosEnv):
    def __init__(self,
                 img_in_state,
                 depth_in_state,
                 dense_reward,
                 grip_in_action,
                 default_grip_state,
                 num_objs,  # number of objects that can be interacted with
                 robot_config_file=None,  # yaml config file
                 state_data=('pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'force_torque',
                             'timestep'),
                 valid_act_t_dof=(1, 1, 1),
                 valid_act_r_dof=(1, 1, 1),
                 max_real_time=10,  # in seconds
                 success_causes_done=False,
                 failure_causes_done=False,
                 reset_teleop_available=False,
                 success_feedback_available=False,
                 obj_reset_box_size=(.025, .025),
                 obj_needs_resetting=True  # for simple state reaching envs, goal can just be arbitrary pos in space
                 ):
        super().__init__(img_in_state=img_in_state,
                         depth_in_state=depth_in_state,
                         dense_reward=dense_reward,
                         grip_in_action=grip_in_action,
                         default_grip_state=default_grip_state,  # 'o' for open, 'c' for closed
                         num_objs=num_objs,  # number of objects that can be interacted with
                         robot_config_file=robot_config_file,  # yaml config file
                         state_data=state_data,
                         valid_act_t_dof=valid_act_t_dof,
                         valid_act_r_dof=valid_act_r_dof,
                         max_real_time=max_real_time,  # in seconds
                         success_causes_done=success_causes_done,
                         failure_causes_done=failure_causes_done,
                         reset_teleop_available=reset_teleop_available,
                         success_feedback_available=success_feedback_available)

        self.obj_reset_box_size = obj_reset_box_size
        self.obj_needs_resetting = obj_needs_resetting

    def _reset_helper(self):
        if self.obj_needs_resetting:
            reset_highs = np.tile(np.array(list(self.obj_reset_box_size) + [360]), [self.num_objs, 1])
            new_obj_pos = self.np_random.uniform(np.zeros([self.num_objs, 3]), reset_highs)
            print_str = "New obj pos: [%.3f, %.3f], rot: %.0f. \n" % (*new_obj_pos[0],)
            if len(new_obj_pos > 1):
                for pos in new_obj_pos[1:]:
                    print_str += "             [%.3f, %.3f], rot: %.0f. \n" % (*pos,)
            if self._reset_teleop_available:
                print(print_str + " Press controller reset button when finished resetting.")
            else:
                input(print_str + " Press enter when finished resetting.")
        else:
            pass