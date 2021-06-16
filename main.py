from stable_baselines3.ppo.ppo import PPO
import torch.nn as nn
import gym
import os
from panda import PandaEnv
import pybullet as p
from pyBullet import PyBullet
from task.Grasp.PandaGraspEnv import PandaGraspEnv
# from PandaReachEnv import PandaReachEnv
from task.wrapper import MoveConstraint, ProcessFrame84, ImageToPyTorch, ProcessDepthFrame84, DepthToPyTorch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import numpy as np
from custom_policy import CustomCNN, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from callback import CheckpointCallback, EvalCallback, SaveVecNormalizeCallback
env = PandaGraspEnv(sim = PyBullet(render =True))

env = ProcessFrame84(env)
env = ImageToPyTorch(env)
env = MoveConstraint(env)

'''
env = ProcessDepthFrame84(env)
env = DepthToPyTorch(env)
'''
env.reset()
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./log/',
                                         name_prefix='rl_model')
save_vec_normalize = SaveVecNormalizeCallback(
                save_freq=1, save_path='./log/Vec/')
eval_callback = EvalCallback(
                eval_env=env,
                callback_on_new_best=save_vec_normalize,
                best_model_save_path='./log/best_model/',
                n_eval_episodes=5,
                log_path='./log/Eval/',
                eval_freq=10000,
                deterministic=True,
            )
model = PPO(CustomActorCriticPolicy,env=env, verbose=1, tensorboard_log='/tmp/ppo/')
# model.learn(10000, callback = checkpoint_callback)
