
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
'''
env = PandaGraspEnvSim(sim = PyBullet(render =True))

env = ProcessGrayFrame84(env)
env = GrayToPyTorch(env)
env = MoveConstraint(env)
'''

env = PandaGraspEnv(sim = PyBullet(render =True))

env = ProcessFrame84(env)
env = ImageToPyTorch(env)
env = MoveConstraint(env)

image = env.reset()
print(env.action_space)

# plt.figure()
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title('Example extracted screen')
# plt.show()
policy_kwargs = dict(
    features_extractor_class=ResNetNetwork,
    features_extractor_kwargs=dict(features_dim=64),
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

model = PPO('MlpPolicy', env=env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(200, callback = checkpoint_callback)
print("Learning Finished")