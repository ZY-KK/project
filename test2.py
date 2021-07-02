
import torch.nn as nn
import gym
import os
from panda import PandaEnv
from bullet.pyBullet import PyBullet
from task.Grasp.PandaGraspEnv import PandaGraspEnv
from task.Reach.PandaReachEnv import PandaReachEnv
from task.wrapper import ProcessGrayFrame64, ProcessFrame64, DepthToPyTorch,GrayToPyTorch, ImageToPyTorch, MoveConstraint,ProcessDepthFrame64, ProcessFrame84
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

env = gym.make('PandaGraspEnv_color-v0')
env = ProcessFrame64(env)
env = ImageToPyTorch(env)

image = env.reset()
# print(image.shape)
# plt.figure()
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title('Example extracted screen')
# plt.show()
# time.sleep(2)

# image = env.reset()
# plt.figure()
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title('Example extracted screen')
# plt.show()
# print("222222222")
# env.step([0.00, 0.00, 0.00, -0.9])
time.sleep(2)
for _ in range(1000):
            env.step([0.00, 0.00, -0.02, 0.10])
            if env.check_contact_plane():
                env.step([0.00, 0.00, -0.02, -0.10])
                grasp = env.get_grasped_object()
                if len(grasp)!=0:
                    for _ in range(1000):
                        env.step([0.00, 0.00, 0.02, -0.10])

                # l_p = env.get_contact_points_left()
                # print(l_p.shape)