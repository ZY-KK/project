
import torch.nn as nn
import gym
import os
from panda import PandaEnv
from bullet.pyBullet import PyBullet
from task.Grasp.PandaGraspEnv import PandaGraspEnv
from task.Reach.PandaReachEnv import PandaReachEnv
from task.wrapper import ProcessGrayFrame84, ProcessFrame84, GrayToPyTorch, ImageToPyTorch, MoveConstraint
# from PandaReachEnv import PandaReachEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import numpy as np
from custom_policy import CustomCNN, CustomActorCriticPolicy, ResNetNetwork
from stable_baselines3.common.evaluation import evaluate_policy
from callback import CheckpointCallback, EvalCallback, SaveVecNormalizeCallback
import time

env = gym.make('PandaGraspEnv-v0')
env = ProcessFrame84(env)
env = ImageToPyTorch(env)

env.reset()
time.sleep(2)
print("222222222")
env.step([0.00, 0.00, -0.02, -0.5])
time.sleep(2)
for _ in range(1000):
            env.step([0.00, 0.00, -0.02, 0.10])
            if env.check_contact_plane():
                env.step([0.00, 0.00, -0.02, -0.10])
                grasp = env.get_grasped_object()
                if len(grasp)!=0:
                    for _ in range(1000):
                        env.step([0.00, 0.00, 0.02, -0.10])