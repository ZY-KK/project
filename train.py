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
	
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment and its parameters
    parser.add_argument("--env", type=str,
                        default="PandaGraspEnv-v0",
                        help="environment ID")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict,
                        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"],
                        default="dummy",
                        help="VecEnv type")

    # Algorithm
    parser.add_argument("--algo", type=str, choices=list(ALGOS.keys()), required=False,
                        default="sac", help="RL Algorithm")
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("--num-threads", type=int,
                        default=-1,
                        help="Number of threads for PyTorch (-1 to use default)")

    # Training duration
    parser.add_argument("-n", "--n-timesteps", type=int,
                        default=-1,
                        help="Overwrite the number of timesteps")

    # Continue training an already trained agent
    parser.add_argument("-i", "--trained-agent", type=str,
                        default="",
                        help="Path to a pretrained agent to continue training")

    # Random seed
    parser.add_argument("--seed", type=int,
                        default=-1,
                        help="Random generator seed")

    # Saving of model
    parser.add_argument("--save-freq", type=int,
                        default=10000,
                        help="Save the model every n steps (if negative, no checkpoint)")
    parser.add_argument("--save-replay-buffer", action="store_true",
                        default=False,
                        help="Save the replay buffer too (when applicable)")

    # Pre-load a replay buffer and start training on it
    parser.add_argument("--preload-replay-buffer", type=str,
                        default="",
                        help="Path to a replay buffer that should be preloaded before starting the training process")

    # Logging
    parser.add_argument("-f", "--log-folder", type=str,
                        default="logs",
                        help="Log folder")
    parser.add_argument("-tb", "--tensorboard-log", type=str,
                        default="tensorboard_logs",
                        help="Tensorboard log dir")
    parser.add_argument("--log-interval", type=int,
                        default=-1,
                        help="Override log interval (default: -1, no change)")
    parser.add_argument("-uuid", "--uuid", action="store_true",
                        default=False,
                        help="Ensure that the run has a unique ID")

    # Hyperparameter optimization
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true",
                        default=False,
                        help="Run hyperparameters search")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "skopt"],
                        default="tpe",
                        help="Sampler to use when optimizing hyperparameters")
    parser.add_argument("--pruner", type=str, choices=["halving", "median", "none"],
                        default="median",
                        help="Pruner to use when optimizing hyperparameters")
    parser.add_argument("--n-trials", type=int,
                        default=10,
                        help="Number of trials for optimizing hyperparameters")
    parser.add_argument("--n-startup-trials", type=int,
                        default=5,
                        help="Number of trials before using optuna sampler")
    parser.add_argument("--n-evaluations", type=int,
                        default=2,
                        help="Number of evaluations for hyperparameter optimization")
    parser.add_argument("--n-jobs", type=int,
                        default=1,
                        help="Number of parallel jobs when optimizing hyperparameters")
    parser.add_argument("--storage", type=str,
                        default=None,
                        help="Database storage path if distributed optimization should be used")
    parser.add_argument("--study-name", type=str,
                        default=None,
                        help="Study name for distributed optimization")

    # Evaluation
    parser.add_argument("--eval-freq", type=int,
                        default=-1,
                        help="Evaluate the agent every n steps (if negative, no evaluation)")
    parser.add_argument("--eval-episodes", type=int,
                        default=5,
                        help="Number of episodes to use for evaluation")

    # Verbosity
    parser.add_argument("--verbose", type=int,
                        default=1,
                        help="Verbose mode (0: no output, 1: INFO)")

    # HER specifics
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )

    args = parser.parse_args()

    train(args=args)

