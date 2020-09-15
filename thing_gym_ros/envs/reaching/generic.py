from thing_gym_ros.envs.thing_ros_generic import ThingRosEnv


class ThingRosReachingGeneric(ThingRosEnv):
    def __init__(self,
                 img_in_state,
                 depth_in_state,
                 dense_reward,
                 grip_in_action,
                 num_objs,  # number of objects that can be interacted with
                 robot_config_file=None,  # yaml config file
                 state_data=('pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot'),
                 valid_act_t_dof=(1, 1, 1),
                 valid_act_r_dof=(1, 1, 1),
                 max_real_time=5,  # in seconds
                 success_causes_done=False,
                 failure_causes_done=False,
                 reset_teleop_available=False,
                 success_feedback_available=False,
                 obj_reset_box_size=(.1, .1),
                 obj_needs_resetting=True  # for simple state reaching envs, goal can just be arbitrary pos in space
                 ):
        super().__init__(img_in_state=img_in_state,
                         depth_in_state=depth_in_state,
                         dense_reward=dense_reward,
                         grip_in_action=grip_in_action,
                         default_grip_state='o',  # 'o' for open, 'c' for closed
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
            new_obj_pos = self.np_random.uniform([0, 0], self.obj_reset_box_size)
            new_obj_rot = self.np_random.uniform(0, 360)
            print_str = "New obj pos: [%.3f, %.3f], rot: %.0f." % \
                        (new_obj_pos[0], new_obj_pos[1], new_obj_rot)
            if self._reset_teleop_available:
                print(print_str + " Press controller reset button when finished resetting.")
            else:
                input(print_str + " Press enter when finished resetting.")
        else:
            pass

# class ThingRosReachingMBGeneric(ThingRosReachingGeneric, thing_ros_mb_generic.ThingRosMBEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)