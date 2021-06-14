from stable_baselines3.ppo.ppo import PPO
import torch.nn as nn
import gym
import os
from panda import PandaEnv
import pybullet as p
from pyBullet import PyBullet
from task.Grasp.PandaGraspEnv import PandaGraspEnv
# from PandaReachEnv import PandaReachEnv
from task.wrapper import ProcessFrame84, ImageToPyTorch, MoveTowardZ, ProcessDepthFrame84, DepthToPyTorch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import numpy as np
from custom_policy import CustomCNN, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import time
env = PandaGraspEnv(sim = PyBullet(render =True))

env = ProcessFrame84(env)
#env = ImageToPyTorch(env)
env = MoveTowardZ(env)
'''
env = ProcessDepthFrame84(env)
env = DepthToPyTorch(env)
'''
env.reset()
for i in range(1000):
    action = env.action_space.sample()
    print(action)
    obs, _, _, _ = env.step(action)
    # print(obs)
    time.sleep(1)