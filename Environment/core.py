import gym
from gym import utils, spaces
import numpy as np


class PyBulletRobot:


    def __init__(self, sim, body_name, file_name, base_position) -> None:
        self.sim = sim
        self.body_name = body_name
        
        self.load_robot(file_name, base_position)
        self.setup()

        

    def load_robot(self,file_name, base_position):
        self.sim.loadURDF(
            body_name = self.body_name,
            fileName = file_name,
            basePosition = base_position,
            useFixedBase = True
        )

    def setup(self):
        """Called once in en constructor."""
        pass

    def set_action(self, action):
        """Perform the action."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the robot."""
        raise NotImplementedError

    def reset(self):
        """Reset the robot."""
        raise NotImplementedError

    def get_link_position(self, link):
        """Returns the position of a link as (x, y, z)"""
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link):
        """Returns the velocity of a link as (vx, vy, vz)"""
        return self.sim.get_link_velocity(self.body_name, link)

    def control_joints(self, target_angles):
        """Control the joints of the robot."""
        self.sim.control_joints(
            body=self.body_name,
            joints=self.JOINT_INDICES,
            target_angles=target_angles,
            forces=self.JOINT_FORCES,
        )
    def get_inverse_kinematics(self, link, newPos, orientation):
        jointPos = self.sim.inverse_kinematics(self.body_name,link, newPos, orientation)

        return jointPos
