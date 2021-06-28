
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
from .core import PyBulletRobot
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
    rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
    def __init__(self, sim, base_position, fingers_friction=5.0):
        self.ee_link = 11
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
        )
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[0], fingers_friction)
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[1], fingers_friction)
        self.create_space()
        # True: open, False: close
        self.gripper_state = True
        self.block_gripper = False

    def gripper_close(self):
        # left close
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[0], 0.0)

        # right close
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[1], 0.0)
        self.gripper_state = False
    def gripper_control(self, value):
        # left open
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[0], value)
        # right open
        self.sim.set_joint_angle(self.body_name, self.FINGERS_INDICES[1], value)

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
        # observation_space = self.create_observation_space()

        return action_space

    def create_observation_space():
        pass

    def create_action_space(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,))
        
    '''  
    def set_action(self, action):
        current_pos = self.get_ee_position()
        gripper_action = action[3]
        
        if gripper_action>0:
            self.gripper_open()
            self.gripper_state =True
        elif gripper_action<0:
            self.gripper_close()
            self.gripper_state=False
        
        self.gripper_control(action[3])
        pos = action[0:3]
        
        newPos = [
                    current_pos[0]+pos[0],
                    current_pos[1]+pos[1],
                    current_pos[2]+pos[2]    
                    
                ]
        
        # newPos = action[0:3]
        orientation = action[4:8]
        # print('action:', action)
        # TODO apply action
        angles = self.get_inverse_kinematics(newPos, orientation)
        # angles = self.get_inverse_kinematics([0.5, -0.1, 0.04], orientation)
        self.set_joint_values(angles)
    '''
    def set_action(self, action):
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_ctrl = action[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_ctrl
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = max(0.63, target_ee_position[2])

        # compute the new joint angles
        orientation = self.sim.get_quaternion_from_euler([0.,-np.pi,np.pi/2.])
        target_angles = self.inverse_kinematics(
            position=target_ee_position, orientation=orientation
        )
        
        if not self.block_gripper:
            fingers_ctrl = action[3] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
            target_angles[-2:] = [target_fingers_width / 2, target_fingers_width / 2]
        
        
        self.control_joints(target_angles=target_angles)
        '''
        if action[3]>0:
            self.gripper_open()
        else:
            self.gripper_close()
        '''
    
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
        self.set_joint_values(self.rest_poses)

    def set_joint_values(self, angles):
        """Set the joint position of a body. Can induce collisions.
        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(
            self.body_name, joints=self.JOINT_INDICES, angles=angles
        )

    def inverse_kinematics(self, position, orientation):
        """Compute the inverse kinematics and return the new joint values. The last two
        coordinates (fingers) are [0, 0].

        Args:
            position (x, y, z): Desired position of the end-effector.
            orientation (x, y, z, w): Desired orientation of the end-effector.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(
            self.body_name, ee_link=11, position=position, orientation=orientation
        )
        # Replace the fingers coef by [0, 0]
        inverse_kinematics = list(inverse_kinematics[0:7]) + [0, 0]
        return inverse_kinematics
    def get_fingers_width(self):
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[1])
        return finger1 + finger2