from __future__ import annotations
from stable_baseline3.common import logger
from typing import Union, Type, Dict
import enum
import math

@enum.unique
class GraspStep(enum.Enum):
    

    REACH = 1
    TOUCH = 2
    GRAPS = 3
    LIFT  = 4

    @classmethod
    def first(self) -> Type[GraspStep]:

        return GraspStep(1)

    def last(self) -> Type[GraspStep]:
        
        length = len(GraspStep)

        return GraspStep(length)
    def next(self) -> Union[Type[GraspStep], None]:

        next_val = self.value+1

        if next_val>self.last().value:
            return None

        else:
            return GraspStep(next_val)

    
class Curriculum():
    def __init__(self, task, enable_ws_scale: bool, min_scale:float, enable_object_increase: bool, max_count:int, enable_steps:bool, reach_required_dis:float, from_reach: bool, sparse_reward:bool) -> None:

        self.enable_steps = enable_steps
        
        self.from_reach =from_reach
        self.sparse_reward = sparse_reward
        # make sure that the current begin step [REACH, TOUCH, GRASP, LIFT]

        self.reach_required_dis = reach_required_dis
        if not self.enable_stages:
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


        kwargs = []
        object_ids = self.task.get_object_ids()
        pos_tmp = {}
        for obj in object_ids:
            pos_tmp[obj]=tuple(self.task.get_target_pos(obj))


        kwargs['object_pos'] = pos_tmp

        # TODO for loop


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

        
            





