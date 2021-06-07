import gym
from gym import error, spaces, utils
from gym.utils import seeding
from robot.panda import Panda
import os
from numpy.core.fromnumeric import shape
from typing import Tuple, List, Union, Dict
import sys
import pybullet_data
import math
import numpy as np
import random
import cv2

class PandaGraspEnv(gym.env):

    def __init__(self, sim) -> None:
        super().__init__()
        self.n_action = 10
        self.sim = sim
        self._is_done = False
        self.robot = Panda(self.sim, base_position = [0.0, 0.0, 0.0])
        self.object_ids = []

    def create_action_space(self):
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=(self.n_action,), dtype = np.float32)

        return self.action_space

    def is_done(self):
        return self._is_done

    def get_observation(self):
        depth_array = self.sim.render(mode = 'depth_array')

        

    def get_contact_points_left(self):
        bodyA = self.robot
        linkIndexA = 9 # 9, 10 
        contact_points = self.sim.get_contact_points(bodyA=bodyA, linkIndexA = linkIndexA)

        return contact_points

    def get_contact_points_right(self):
        bodyA = self.robot
        linkIndexA = 10 # 9, 10 
        contact_points = self.sim.get_contact_points(bodyA=bodyA, linkIndexA = linkIndexA)
        return contact_points
    def get_grasped_object(self):
        # if gripper open
        grasp_model_ready = {}
        if self.robot.get_gripper_state==True:
            return []
        contact_points_left = self.get_contact_points_left()
        contact_points_right = self.get_contact_points_right()
        if len(contact_points_left==0):
            return []
        else:



    def get_closest_object_dis(self, object_positions: Dict[str, Tuple[float, float, float]]):
        min_distance = sys.float_info.max

        ee_position = self.robot.get_ee_position()
        for object_position in object_positions.values():
            distance = np.linalg.norm([ee_position[0] - object_position[0],
                                       ee_position[1] - object_position[1],
                                       ee_position[2] - object_position[2]])
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_distance_obj(self, target_position, ee_position):
        return np.linalg.norm([ee_position[0] - target_position[0],
                               ee_position[1] - target_position[1],
                               ee_position[2] - target_position[2]])

    def get_target_orientation(self, objectUid):
        _, orientation = self.sim.get_object_pos_and_orientation(objectUid)
        return orientation

    def get_target_pos(self, objectUid):
        pos, _ = self.sim.get_object_pos_and_orientation(objectUid)
        return pos
        

    def get_inverse_kinematics(self, newPos, orientation):
        return self.robot.get_inverse_kinematics(newPos, orientation)
    def get_object_ids(self):
        return self.object_ids
    def _create_scene(self):
        self.sim.add_plane(basePosition = [0, 0, -0.65])
        self.sim.add_table(basePosition = [0.5,0,-0.65])
        self.object_000_id = self.sim.add_object_000([0.65, 0, 0])
        self.object_ids.append(self.object_000_id)
        