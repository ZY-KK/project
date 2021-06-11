from stable_baselines3.ppo.ppo import PPO
import torch.nn as nn
import gym
import os
from panda import PandaEnv
import pybullet as p
from pyBullet import PyBullet
from task.Grasp.PandaGraspEnv import PandaGraspEnv
# from PandaReachEnv import PandaReachEnv
from task.wrapper import ProcessFrame84, ImageToPyTorch, MoveTowardZ
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import numpy as np
from custom_policy import CustomCNN, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
env = PandaGraspEnv(sim = PyBullet(render =True))

env = ProcessFrame84(env)
env = ImageToPyTorch(env)

for _ in range(50):
    action = env.action_space.sample()
    print(action)
    env.step(action)

env.close()
