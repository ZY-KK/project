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
    def __init__(self, task, enable_ws_scale: bool, min_scale:float, enable_object_increase: bool, max_count:int, enable_steps:bool, ) -> None:

        self.enable_steps = enable_steps
        


        if not self.enable_stages:
            self.step: GraspStep = GraspStep.last()


        self.step_completed: Dict[GraspStep, bool] = {
            GraspStep(step): False for step in range(GraspStep.first().value, GraspStep.last().value+1)}
        
        self.success_rate = 0.0
        self.is_sucess = False
        self.is_failure = False

    def get_reward(self)->float:

        reward = 0.0

        first_step = GraspStep.last().value
        # check if which step is not completed
        for step  in range(GraspStep.first().value, GraspStep.last().value+1):
            if not self.step_completed[GraspStep(step)]:
                first_step = step
                break

        