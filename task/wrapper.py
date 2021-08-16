from math import inf
import gym
import numpy as np
import cv2
from gym.wrappers import TimeLimit
from numpy.core.fromnumeric import shape

WIDTH = 64
HEIGHT = 64
class ProcessFrame64(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame64, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame64.process(self.env.sim.render(mode='rgb_array'))

    def process(frame):
        if frame.size == 720 * 960 * 3:
            img = np.reshape(frame, [720, 960, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        #img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #      img[:, :, 2] * 0.114

        resized_screen = cv2.resize(
            img, (112, HEIGHT), interpolation=cv2.INTER_AREA)
        y_t = resized_screen[:, 24:88]
        
        y_t = np.reshape(y_t, [WIDTH, HEIGHT, 3])
        return y_t.astype(np.uint8)
class ProcessGrayFrame64(gym.ObservationWrapper):
    def __init__(self, env: None) -> None:
        super(ProcessGrayFrame64, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (64, 64, 1))
    def observation(self, obs):
        return ProcessGrayFrame64.process(self.sim.render(mode= 'rgb_array'))
    def process(frame):
        # print('frame',frame)
        if frame.size == 720*960*3:
            img = np.reshape(frame, [720, 960, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (112, 64), interpolation=cv2.INTER_AREA)
        y_t = resized_screen[:, 24:88]
        
        y_t = np.reshape(y_t, [64, 64, 1])
        return y_t.astype(np.uint8)

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(self.env.sim.render(mode='rgb_array'))

    def process(frame):
        if frame.size == 720 * 960 * 3:
            img = np.reshape(frame, [720, 960, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
             img[:, :, 2] * 0.114

        resized_screen = cv2.resize(
            img, (112, 84), interpolation=cv2.INTER_AREA)
        y_t = resized_screen[:, 14:98]
        
        y_t = np.reshape(y_t, [84, 84, 1])
        return y_t.astype(np.uint8)

class GrayToPyTorch(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super(GrayToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low = 0.0, high = 255.0, shape = new_shape, dtype = np.float32)
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class MoveConstraint(gym.ActionWrapper):
    def __init__(self, env):
        super(MoveConstraint, self).__init__(env)
        self.constraint = [0.1, 0.5]
        self.env = env
    def action(self, action):
        action[2] = -.1
        x = np.clip

        orientation = self.env.get_quaternion_from_euler([0.,-np.pi,np.pi/2.])
        return action
class ProcessDepthFrame64(gym.ObservationWrapper):
    def __init__(self,env:None) -> None:
        super(ProcessDepthFrame64, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low = 0, high = 1.0, shape = (64, 64, 1))
    def observation(self, obs):
        return ProcessDepthFrame64.process(self.sim.render(mode= 'depth_array'))
    def process(frame):
        # print('frame:  ',frame.size)
        if frame.size == 720*960*1:
            img = np.reshape(frame, [720, 960, 1]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #       img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (112, 64), interpolation=cv2.INTER_AREA)
        y_t = resized_screen[:, 24:88]
        
        y_t = np.reshape(y_t, [64, 64, 1])
        return y_t.astype(np.float32)

class DepthToPyTorch(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super(DepthToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low = 0.0, high = 1.0, shape = new_shape, dtype = np.float32)
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        # ic("time reset")
        return self.env.reset(**kwargs)
        
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))





class ObservationForPosition(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationForPosition).__init__(env)
        self.env = env
        self.observation_space = gym.space.Box(low = -np.inf, high = np.inf, shape=(6, ), dtype = np.float32)
        



__all__ = ["Monitor", "ResultsWriter", "get_monitor_files", "load_results"]

import csv
import json
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from icecream import ic
class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "sac_monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is not None:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords,
            )
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = False
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        # ic("wrapper_reset")
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        '''
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        '''
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        
        self.needs_reset = True
        ep_rew = sum(self.rewards)
        ep_len = len(self.rewards)
        ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
        for key in self.info_keywords:
            ep_info[key] = info[key]
        self.episode_returns.append(ep_rew)
        self.episode_lengths.append(ep_len)
        self.episode_times.append(time.time() - self.t_start)
        ep_info.update(self.current_reset_info)
        if self.results_writer:
            self.results_writer.write_row(ep_info)
        info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return:
        """
        return self.episode_times


class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """

    pass


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class
    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None,
        extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.file_handler = open(filename, "wt")
        self.file_handler.write("#%s\n" % json.dumps(header))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler
        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()


def get_monitor_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path
    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, "*" + Monitor.EXT))


def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``
    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame
