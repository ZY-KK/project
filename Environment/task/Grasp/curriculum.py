from __future__ import annotations
from stable_baselines3.common import logger
from typing import Union, Type, Dict
import enum
import math

@enum.unique
class GraspStep(enum.Enum):
    

    REACH = 1
    TOUCH = 2
    GRASP = 3
    LIFT  = 4
    @classmethod
    def first(self) -> Type[GraspStep]:

        return GraspStep(1)
    @classmethod
    def last(self) -> Type[GraspStep]:
        
        length = len(GraspStep)
        return GraspStep(length)
    @classmethod
    def next(self) -> Union[Type[GraspStep], None]:

        next_val = self.value+1

        if next_val>self.last().value:
            return None

        else:
            return GraspStep(next_val)

    
class Curriculum():
    def __init__(self, task, enable_ws_scale: bool, min_scale:float, enable_object_increase: bool, max_count:int, enable_steps:bool, reach_required_dis:float, from_reach: bool, sparse_reward:bool,step_increase_rewards: bool,step_reward_multiplier:float, verbose: bool, lift_height:float,act_quick_reward:float, ground_collision_reward:float, ground_collisions_till_termination:int, success_rate_rolling_average_n:int, restart_every_n_steps:int, success_rate_threshold: float, restart_exploration:bool) -> None:

        self.enable_steps = enable_steps
        
        self.from_reach =from_reach
        self.sparse_reward = sparse_reward
        # make sure that the current begin step [REACH, TOUCH, GRASP, LIFT]
        self.task = task
        self.reach_required_dis = reach_required_dis
        self.step_increase_rewards = step_increase_rewards
        self.step_reward_multiplier = step_reward_multiplier
        self.verbose = verbose
        self.lift_height= lift_height
        self.act_quick_reward = act_quick_reward
        self.ground_collision_reward = ground_collision_reward
        self.grond_collision_counter = 0
        self.ground_collisions_till_termination = ground_collisions_till_termination
        self.normalize_positive_reward_multiplier = 1.0
        self._normalize_negative_reward_multiplier =1.0
        self.success_rate = 0.0
        self.success_rate_rolling_average_n =success_rate_rolling_average_n
        self.restart_every_n_steps =restart_every_n_steps
        self.reset_step_counter = restart_every_n_steps
        self.success_rate_threshold = success_rate_threshold
        if not self.enable_steps:
            self.step: GraspStep = GraspStep.last()
        elif self.from_reach:
            self.step:GraspStep = GraspStep.REACH
        else:
            self.step:GraspStep = GraspStep.TOUCH
        

        self.step_completed: Dict[GraspStep, bool] = {
            GraspStep(step): False for step in range(GraspStep.first().value, GraspStep.last().value+1)}
        
        self.success_rate = 0.0
        self.is_sucess = False
        self.is_failure = False
        self.previous_min_dis = 0.0
    def get_reward(self)->float:

        reward = 0.0

        first_step = GraspStep.last().value
        # check if which step is not completed
        for step  in range(GraspStep.first().value, GraspStep.last().value+1):
            if not self.step_completed[GraspStep(step)]:
                first_step = step
                break


        kwargs = {}
        object_ids = self.task.get_object_ids()
        pos_tmp = {}
        for obj in object_ids:
            pos_tmp[obj]=tuple(self.task.get_target_pos(obj))


        kwargs['object_pos'] = pos_tmp

        # TODO for loop
        for step in range(first_step, GraspStep.last().value+1):
            if step>=GraspStep.GRASP.value and not 'grasp_obj' in kwargs:
                kwargs['grasp_obj'] =self.task.get_grasped_object()
            if self.step_increase_rewards:
                reward_factor = self.step_reward_multiplier**(step-1)
            else:
                reward_factor = self.step_reward_multiplier**(self.step.value-step)
            # TODO: reward
            reward+=reward_factor*self.GET_REWARD[GraspStep(step)](self, **kwargs)

            if not self.step_completed[GraspStep(step)]:
                break
        reward*=self.normalize_positive_reward_multiplier
        neg_reward = self.reward_all(**kwargs)
        neg_reward*=self._normalize_negative_reward_multiplier

        reward+=neg_reward

        return reward
    def update_success_rate(self, is_success:bool):
        self.success_rate = ((self.success_rate_rolling_average_n-1)*self.success_rate+float(is_success))/self.success_rate_rolling_average_n

        if self.verbose:
            print('')
        

    def is_done(self):
        if self.is_sucess:
            self.update_success_rate(is_success=True)
            return True
        elif self.is_failure:
            self.update_success_rate(is_success=False)
            return True
        else:
            return False
    def check_restart_first(self):

        if self.reset_step_counter<=0:
            

            self.step = GraspStep.first()

            self.success_rate = 0.5*self.success_rate_threshold
            self.reset_step_counter = self.restart_every_n_steps
            self.restart_exploration = True
            return True

    def get_info(self):
        if self.step!=GraspStep.first() and self.restart_every_n_steps>0:
            self.reset_step_counter-=1
            self.check_restart_first()
            info = {'is_success': self.step_completed[GraspStep.last()],
                'curriculum.restart_exploration': self.restart_exploration}
            self.restart_exploration=False

            return info
    def next_step(self):
        if self.step ==GraspStep.last():
            return False
        if self.success_rate>self.success_rate_threshold:
            next = self.step.next()
            self.step = next

            self.success_rate =0.0
            self.restart_exploration  =True
            return True
        return False
    def reset_task(self):

        if not (self.is_sucess or self.is_failure):
            self.update_success_rate(is_success=False)
        self.next_step()
        self._log_curriculum()

        self.is_sucess = False
        self.is_failure = False

        for step in range(GraspStep.first().value, GraspStep.last().value+1):
            self.step_completed[GraspStep(step)]=False
        self.grond_collision_counter = 0

    def reward_reach(self, **kwargs)-> float:

        object_pos = kwargs['object_pos']
        min_distance  = self.task.get_closest_object_dis(object_pos)
        if min_distance<self.reach_required_dis:
            if GraspStep.REACH.value>=self.step:
                self.is_sucess = True
            self.step_completed[GraspStep.REACH] = True

            if self.sparse_reward:
                return 1.0
        
        if self.sparse_reward:
            return 0.0
        else:
            difference = self.previous_min_dis-min_distance
            self.previous_min_dis = min_distance

            #TODO dence reward
            


    def reward_touch(self, **kwargs) -> float:
        contact_points_left = self.task.get_contact_points_left()
        contact_points_right = self.task_get_contact_points_right()
        if len(contact_points_left)>0 or len(contact_points_right)>0:
            if GraspStep.TOUCH.value>=self.step:
                self.is_sucess = True

            self.step_completed[GraspStep.TOUCH] = True
            return 1.0
        else:
            return 0.0
            
    def reward_grasp(self, **kwargs)-> float:
        grasp_obj = kwargs['grasp_obj']
        if len(grasp_obj)>0:
            if GraspStep.GRASP.value>=self.step:
                self.is_sucess = True
            else:
                self.is_sucess
            if self.verbose:
                print('grasp_obj', grasp_obj)
        
            return 1.0
        else:
            return 0.0
        
    def reward_lift(self, **kwargs):
        grasp_obj = kwargs['grasp_obj']
        if len(grasp_obj)==0:
            return 0.0
        reward = 0.0
        obj_pos = kwargs['object_pos']
        for obj in grasp_obj:
            if obj_pos[obj][2]>self.lift_height:
                if GraspStep.LIFT.value>=self.step.value:
                    self.is_sucess = True
                else:
                    self.is_sucess
                self.step_completed[GraspStep.LIFT]=True
                if self.sparse_reward:
                    reward+=1.0
                # TODO: not sparse reward

    def reward_all(self, **kwargs):

        reward = -self.act_quick_reward
        if self.task.check_contact_plane():
            reward-=self.ground_collision_reward
            self.grond_collision_counter+=1
            if self.grond_collision_counter>=self.ground_collisions_till_termination:
                self.is_failure=True
            return reward
        






            
    def _log_curriculum(self):
        logger.record("curriculum/current_stage",
                      self.step, exclude="tensorboard")
        logger.record("curriculum/current_stage_id",
                      self.step.value, exclude="stdout")
        logger.record("curriculum/current_success_rate",
                      self.success_rate)
        if self.restart_every_n_steps > 0:
            logger.record("curriculum/steps_until_reset",
                          self.reset_step_counter)

    GET_REWARD = {
        GraspStep.REACH : reward_reach,
        GraspStep.TOUCH : reward_touch,
        GraspStep.GRASP :reward_grasp,
        GraspStep.LIFT : reward_lift,
    }




