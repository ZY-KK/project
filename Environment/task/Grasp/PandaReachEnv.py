
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from robot.panda import Panda
import os
from numpy.core.fromnumeric import shape

import pybullet_data
import math
import numpy as np
import random
import cv2

class PandaGraspEnv(gym.Env):
    
    base_position = [0, 0, 0]

    def __init__(self, sim):
        self.n_action=3
        base_position = [0, 0, 0]
        self.sim = sim
        self._is_done = False
        self._create_scene()
        self.robot = Panda(self.sim, base_position = [0.0, 0.0, 0.0])
        self.create_observation_space()
        self.create_action_space()

    
    def create_action_space(self):
        self.action_space = spaces.Box(low = -1.0, high=1.0, shape=(self.n_action,), dtype=np.float32)

        return self.action_space

    def create_observation_space(self):
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(6, ),dtype=np.float32)

    def step(self,action):
        self.set_action(action)
        obs = self.get_observation()
        reward = self.get_reward()
        info = {'obs': obs, 'reward':reward}
        done  = self.is_done()
        return obs, reward, done, info

    def is_done(self):
        
        return self._is_done
    def reset(self):
        self.robot.reset()
        self._is_done=False
        obs = self.get_observation()
        return obs
    def set_action(self, action):
        # print(f"action: {action}")
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        orientation = self.sim.get_quaternion_from_euler([0.,-math.pi,math.pi/2.])
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        current_pos = self.robot.get_ee_position()
        newPosition = [current_pos[0]+dx, current_pos[1]+dy, current_pos[2]+dz]
        newPosition[2] = max(0, newPosition[2])
        pos = self.get_inverse_kinematics(newPosition, orientation)
        print(f"pos: {pos}")
        self.robot.control_joints(pos)

    def get_reward(self):
        target = self.get_target_pos(self.object_000_id)
        ee_pos = self.robot.get_ee_position()
        dis = self.get_distance_obj(target_position=target, ee_position=ee_pos)
        reward = -dis
        if dis<=0.01:
            self._is_done=True
            if dis < 0.01:
                reward = reward + 1.5
            elif dis < 0.015:
                reward = reward + 1.0
            elif dis < 0.03:
                reward = reward + 0.5
        return reward

    def get_observation(self):
        pass
        
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
        
    def _create_scene(self):
        self.sim.add_plane(basePosition = [0, 0, -0.65])
        self.sim.add_table(basePosition = [0.5,0,-0.65])
        self.object_000_id = self.sim.add_object_000([0.65, 0, 0])


        

        

    