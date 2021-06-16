from numpy.random import seed
from utils import exp_manager
from utils.exp_manager import ExperimentManager
from Environment.task.Grasp import PandaGraspEnv
from utils.utils import ALGOS,StoreDict
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import torch as th
import uuid
import os
import argparse
class Train:
    def __init__(self):
        pass
    def train(args = None):
        if args.seed <0:
            args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()
        set_random_seed(args.seed)
        # Setting num threads to 1 makes things run faster on cpu
        if args.num_threads > 0:
            if args.verbose > 1:
                print(f"Setting torch.num_threads to {args.num_threads}")
            th.set_num_threads(args.num_threads)
        
        if args.trained_agent !='':
            assert args.trained_agent.endswith(".zip") and os.path.isfile(args.trained_agent), "The trained agent should be a .zip file"
        uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
        

        exp_manager = ExperimentManager(
            args,
            algo=args.algo,
            env_id=args.env,
            log_folder=args.log_folder,
            tensorboard_log=args.tensorboard_log,
            n_timesteps=args.n_timestep,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            hyperparams=args.hyperparams,
            env_kwargs=args.env_kwargs,
            trained_agent=args.trained_agent,
            optimize_hyperparameters=args.optimize_hyperparameters,
            storage=args.storage,
            study_name=args.study_name,
            n_trials=args.n_trials,
            n_jobs= args.n_jobs,
            sampler=args.sampler,
            pruner = args.pruner,
            n_startup_trials = args.n_startup_trials,
            n_evaluations = args.n_evaluations,
            truncate_last_trajectory = args.truncate_last_trajectory,
            uuid_str = uuid_str,
            seed = args.seed,
            log_interval=args.log_interval,
            save_replay_buffer=args.save_replay_buffer,
            preload_replay_buffer=args.preload_replay_buffer,
            verbose=args.verbose,
            vec_env_type=args.vec_env,

        )

        model = exp_manager.setup_experiment()

        if args.optimize_hyperparameters:
            exp_manager.hyperparameters_optimization()
        else:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
	
