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
        action[2] = -.3
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
        

    