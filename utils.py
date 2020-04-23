from IPython.display import HTML, clear_output
from IPython import display as ipythondisplay
import glob
import io
import base64
import gym

import tensorflow as tf
import numpy as np


def show_video(path):
    """
    Usage:
    env = gym.wrappers.Monitor(env, "recordings/1", force=True)
    show_video("recordings/1")
    """
    mp4list = glob.glob(path + '/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" 
                    controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def dynamic_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def play_in_video(env, video_path, agent=None, max_steps=None):
    video_env = gym.wrappers.Monitor(env, video_path, force=True)
    state = video_env.reset()
    total_reward = 0
    while max_steps is None or max_steps > 0:
        video_env.render(mode='rgb_array')
        if agent is None:
            action = video_env.action_space.sample()
        else:
            action = agent.next_action(state)
        state, reward, done, info = video_env.step(action)
        total_reward += reward
        if max_steps is not None:
            max_steps -= 1
        if done:
            break
    print("Total reward:", total_reward)
    video_env.close()
    show_video(video_path)


def evaluate(env, agent=None, max_steps=None, times=1):
    total_rewards = []
    for _ in range(times):
        state = env.reset()
        total_reward = 0
        while max_steps is None or max_steps > 0:
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.next_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if max_steps is not None:
                max_steps -= 1
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)
