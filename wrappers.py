import gym
from skimage.transform import resize
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image


def black_and_white(img, size):
    img = img_to_array(array_to_img(img).resize(
        (size[0], size[1]), Image.ANTIALIAS))
    img = img.mean(-1, keepdims=True).astype('float32').reshape(size)/255.
    return img


class ActionMapWrapper(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = action_map

    def action(self, action):
        return self.action_map[action]


class BWrapper(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self.size = size

    def observation(self, observation):
        return black_and_white(observation, self.size)


class EarlyStopWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_negative = 0

    def reset(self, **kwargs):
        self.n_negative = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if reward < 0:
            self.n_negative += 1
            if self.n_negative > 500:
                reward = -5
                is_done = True
        else:
            self.n_negative = 0
        return next_state, reward, is_done, info


class FrameSkipper(gym.Wrapper):
    def __init__(self, env, n_frames, action):
        super().__init__(env)
        self.n_frames = n_frames
        self.action = action

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            state = self.env.step(self.action)[0]
        return state


class MaxStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.n_steps = 0

    def reset(self, **kwargs):
        self.n_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            is_done = True
        return next_state, reward, is_done, info


class StackedLagWrapper(gym.ObservationWrapper):
    def __init__(self, env, lag, concat_axis=-1):
        super().__init__(env)
        self.lag = lag
        self.buffer = []
        self.axis = concat_axis

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.buffer = [state, state, state, state]
        return np.stack(self.buffer, axis=self.axis)

    def observation(self, observation):
        self.buffer = [self.buffer[1], self.buffer[2],
                       self.buffer[3], observation]
        return np.stack(self.buffer, axis=self.axis)
