
from enum import Flag
import os
from numpy.core.fromnumeric import shape
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import cv2
from gym import spaces
from core import PyBulletRobot
class Panda(PyBulletRobot):
    '''
       {
        'panda_link0': -1, 
        'panda_link1': 0, 
        'panda_link2': 1, 
        'panda_link3': 2, 
        'panda_link4': 3, 
        'panda_link5': 4, 
        'panda_link6': 5, 
        'panda_link7': 6, 
        'panda_link8': 7, 
        'panda_hand': 8, 
        'panda_leftfinger': 9, 
        'panda_rightfinger': 10, 
        'panda_grasptarget': 11
        }

    '''
    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    FINGERS_INDICES = [9, 10]
    NEUTRAL_JOINT_VALUES = [0.00, 0.41, 0.00, -1.85, -0.00, 2.26, 0.79, 0, 0]
    JOINT_FORCES = [87, 87, 87, 87, 12, 120, 120, 170, 170]

    def __init__(self, sim, base_position, fingers_friction=1.0):
        self.ee_link = 11
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
        )
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[0], fingers_friction)
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[1], fingers_friction)

        # True: open, False: close
        self.gripper_state = True

    def gripper_close(self):
        # left close
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[0], 0.0)

        # right close
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[1], 0.0)
        self.gripper_state = False
        
    def gripper_open(self):
        # left open
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[0], 0.04)
        # right open
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[1], 0.04)
        self.gripper_state = True

    def get_gripper_state(self):
        return self.gripper_state
        
    def get_ee_position(self):
        return self.get_link_position(self.ee_link)
    def get_ee_velocity(self):
        return self.get_link_velocity(self.ee_link)

    def get_inverse_kinematics(self, newPos, orientation):
        return super().get_inverse_kinematics(self.ee_link, newPos, orientation)

    def create_space(self):
        action_space = self.create_action_space()
        observation_space = self.create_observation_space()

        return action_space, observation_space

    def create_observation_space():
        pass

    def create_action_space(self):

        pass
        
    def set_action(self, action):
        gripper_action = action[0]
        if gripper_action>0:
            self.gripper_open()
            self.gripper_state =True
        elif gripper_action<0:
            self.gripper_close()
            self.gripper_state=False
        pos = action[1:4]
        orientation = action[4:8]
        # TODO apply action
        angles = self.get_inverse_kinematics(pos, orientation)
        self.set_joint_values(angles)
        
        
    
    def get_observation():
        pass

    def get_reward():
        pass
    
    def is_done():
        pass

    def reset(self):
        # self.sim.reset()
        self.set_joint_neutral()

    def set_joint_neutral(self):
        """Set the robot to its neutral pose."""
        self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

    def set_joint_values(self, angles):
        """Set the joint position of a body. Can induce collisions.
        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(
            self.body_name, joints=self.JOINT_INDICES, angles=angles
        )

        