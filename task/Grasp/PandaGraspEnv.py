from .curriculum import Curriculum
import itertools
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
from PIL import Image
from bullet.pyBullet import PyBullet
from icecream import ic
# ic.disable()
class PandaGraspEnv(gym.Env):

    def __init__(self,render) -> None:
        super().__init__()
        self.n_action = 4
        self.sim = PyBullet(render=render)
        self._is_done = False
        '''
        self.robot = Panda(self.sim, base_position = [0.0, 0.0, 0.0])
        self.object_ids = []
        self.robot_id = self.sim.get_body_ids()['panda']
        '''
        
        self.workspace_volum = [0.5, 0.8,-0.2,0.2,0.62,0.75]
        # self.robot_id = self.sim.get_body_ids()['panda']
        # self._create_scene()
        self.curriculum \
            = Curriculum(task=self, enable_ws_scale=False, \
            min_scale=0.1, enable_object_increase=False, max_count=4, \
            enable_steps=True, reach_required_dis=0.05, from_reach=True, \
            sparse_reward=True, step_increase_rewards=True, \
            step_reward_multiplier=7.0, verbose=True, lift_height=0.225, \
            act_quick_reward=0.005, ground_collision_reward=1.0, \
            ground_collisions_till_termination=100, \
            success_rate_rolling_average_n=100, restart_every_n_steps=0, \
            success_rate_threshold=0.7, restart_exploration=False)
        
        self.create_space()
        self._create_scene()
    def create_space(self):
        self.create_observation_space()
        self.create_action_space()

    def create_action_space(self):
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=(self.n_action,), dtype = np.float32)

        return self.action_space
    def create_observation_space(self):
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(6, ),dtype=np.float32)
    

    def get_observation(self):
        '''
        depth_array = self.sim.render(mode = 'depth_array')
        # print("=========test=================")
        return depth_array
        '''
        pass

    def get_reward(self):
        reward = self.curriculum.get_reward()
        return reward

    def is_done(self):
        done =self.curriculum.is_done()
        return done

    def get_info(self):
        info =self.curriculum.get_info()
        # print('info:', info)
        return info
        
    def reset(self):
        
        self.robot.reset()
        self.curriculum.reset_task()
        # self.sim.reset()
        # self._is_done=False
        obs = self.get_observation()
        obs = np.asarray(obs)
        # print('=====', obs)
        # print('reset!!')
        ic('============reset==================')
        return obs

    def get_contact_points_left(self):
        bodyA = self.robot_id
        linkIndexA = 9 # 9, 10 
        contact_points = self.sim.get_contact_points(bodyA=bodyA, linkIndexA = linkIndexA)
        contact_points = np.asarray(contact_points)
        ic(contact_points)
        return contact_points
    def get_contact_points_right(self):
        bodyA = self.robot_id
        linkIndexA = 10 # 9, 10 
        contact_points = self.sim.get_contact_points(bodyA=bodyA, linkIndexA = linkIndexA)
        contact_points = np.asarray(contact_points)
        return contact_points
    def check_outside_ws(self):
        ee_pos = self.robot.get_ee_position()
        if ee_pos[0]<self.workspace_volum[0] or ee_pos[0]>self.workspace_volum[1] \
            or ee_pos[1]<self.workspace_volum[2] or ee_pos[1]>self.workspace_volum[3] \
            or ee_pos[2]<self.workspace_volum[4] or ee_pos[2]>self.workspace_volum[5]:
            return True
        return False

    def step(self, action):
        # TODO step function
        self.sim.step()
        self.robot.set_action(action)
        self.sim.render()
        obs = self.get_observation()
        
        reward = self.get_reward()
        done = self.is_done()
        info = self.get_info()
        return obs, reward, done, info

    
    def get_quaternion_from_euler(self, angle):
        return self.sim.get_quaternion_from_euler(angle)

    def get_grasped_object(self):
        # if gripper open
        grasp_model_ready = {}
        if self.robot.get_gripper_state==True:
            return []
        contact_points_left = self.get_contact_points_left()
        contact_points_right = self.get_contact_points_right()
        if len(contact_points_left)==0:
            return []
        else:
            model_id = contact_points_left[0][2]
            
            for point_l in contact_points_left:
                # print('point_l', point_l)
                if model_id not in grasp_model_ready.keys():
                    grasp_model_ready[model_id] = []
                
                grasp_model_ready[model_id].append(point_l)
        if len(contact_points_right)==0:
            return []
        else:
            model_id = contact_points_right[0][2]
            for point_r in contact_points_right:
                if model_id not in grasp_model_ready.keys():
                    grasp_model_ready[model_id] = []
                
                grasp_model_ready[model_id].append(point_r)
        grasp_objects = []
        # print('grasp_model_ready:', grasp_model_ready)
        for model, points_list in grasp_model_ready.items():
            # the gripper is open, if the num of contact points is less than two, continute
            if len(points_list)<len(self.robot.FINGERS_INDICES):
                continue
            
            normals_avgs = []
            for point in points_list:
                normals_avg = np.array([0.0, 0.0, 0.0])
                # print('points:', point)
                # for point in points:
                    
                #     normals_avg+=point[7] # contactNormalOnB
                normals_avg+=point[7]
                normals_avg/=np.linalg.norm(normals_avg)

                normals_avgs.append(normals_avg)
            # print('normals_avgs',normals_avgs)
            normal_angles = []
            for v_1, v_2 in itertools.combinations(normals_avgs, 2):

                angle = np.arccos(np.clip(np.dot(v_1, v_2), -1.0, 1.0))
                normal_angles.append(angle)

            # print('normal_angles: ',normal_angles)
            angle_threshold = 0.5*np.pi/len(self.robot.FINGERS_INDICES)
            for ag in normal_angles:
                if ag>angle_threshold:
                    grasp_objects.append(model)
                    continue
        return grasp_objects
                
            


                    
                

    


    def get_closest_object_dis(self, object_positions: Dict[str, Tuple[float, float, float]]):
        min_distance = np.inf

        ee_position = self.robot.get_ee_position()
        ic(ee_position)
        for object_position in object_positions.values():
            # print('obj_pos: ', object_position)
            distance = np.linalg.norm([ee_position[0] - object_position[0],
                                    ee_position[1] - object_position[1],
                                    ee_position[2] - object_position[2]])
            ic(distance)
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
        
    def check_contact_plane(self):
        # print('robot:', self.robot_id)
        # print('table:', self.table)
        linkIndexA = 9
        contact_points_l = self.sim.get_contact_points_A_and_B(bodyA = self.robot_id, linkIndexA=linkIndexA, bodyB = self.table)
        linkIndexA = 10
        contact_points_r = self.sim.get_contact_points_A_and_B(bodyA = self.robot_id, linkIndexA=linkIndexA, bodyB = self.table)
        if len(contact_points_l)>0 or len(contact_points_r)>0:
            return True
        return False

    def get_inverse_kinematics(self, newPos, orientation):
        return self.robot.get_inverse_kinematics(newPos, orientation)
    def get_object_ids(self):
        self.object_ids = np.asarray(self.object_ids)
        return self.object_ids
    def get_plane_id(self):
        return self.plane
    def get_table_id(self):
        return self.table
    '''
    def _create_scene(self):
        self.sim.add_plane(basePosition = [0, 0, 0])
        self.plane = self.sim.get_body_ids()['plane']
        # self.sim.add_table(basePosition = [0.5,0,-0.65])
        # self.table = self.sim.get_body_ids()['table']
        self.sim.add_object_000([0.4, 0, 0])
        print('==================')
        self.object_000_id = self.sim.get_body_ids()['000']
        
        self.object_ids.append(self.object_000_id)
        self.add_random_obj()
    '''
    def _create_scene(self):
        
        # self.sim.add_table(basePosition = [0.5,0,-0.65])
        # self.table = self.sim.get_body_ids()['table']
        # self.sim.resetSimulation()
        self.robot = Panda(self.sim, base_position = [0.0, 0.0, 0.6])
        self.object_ids = []
        self.robot_id = self.sim.get_body_ids()['panda']
        # self.workspace_volum = [0.3, 0.3, 0.2]
        self.sim.add_plane(basePosition = [0, 0, 0])
        self.plane = self.sim.get_body_ids()['plane']
        self.sim.add_table(basePosition = [0.5,0,0])
        self.table = self.sim.get_body_ids()['table']
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.65]
        state_object = [0.43, 0.0, 0.65]
        # self.sim.add_object_000(state_object)
        # self.object_000_id = self.sim.get_body_ids()['000']
        # self.object_ids.append(self.object_000_id)
        # self.sim.add_object_020(state_object)
        # self.object_020_id = self.sim.get_body_ids()['020']
        # self.object_ids.append(self.object_020_id)
        # self.sim.add_object_model(state_object)
        # self.object_020_id = self.sim.get_body_ids()['020']
        # self.object_ids.append(self.object_020_id)
        # self.add_random_obj()
        
        self.object_size = 0.04
        self.sim.create_box(
            body_name="object1",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.6, 0.0, self.object_size / 2+0.63],
            rgba_color=[0.9, 0.1, 0.1, 1],
            friction=10,  # increase friction. For some reason, it helps a lot learning
        )
        self.object_object1_id = self.sim.get_body_ids()['object1']
        # print(self.object_object1_id)
        self.object_ids.append(self.object_object1_id)

    def add_random_obj(self):
        # random position
        x = np.random.uniform(low =self.workspace_volum[0], high =self.workspace_volum[1])
        y = np.random.uniform(low = self.workspace_volum[2], high = self.workspace_volum[3])
        z = 0.65
        pos = [x, y, z]
        
        # random id
        obj_id = '000'
        # 578 object not good
        while(obj_id in self.sim.get_body_ids().keys()):
            id = np.random.randint(low = 0, high= 998)
            
            if id<10:
                obj_id = '00'+str(id)
            elif id >=10 and id <100:
                obj_id = '0'+str(id)
            else:
                obj_id = str(id)
            # print('obj_id(env)=', obj_id)
        
        self.sim.add_random_object(obj_id, pos)
        self.object_ids.append(self.sim.get_body_ids()[obj_id])
    

