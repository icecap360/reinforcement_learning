import gymnasium as gym
import cv2
from model import QFunctionConv, MainModel
import torch.nn as nn
import torch
import random 
import numpy as np
import os

class Replay:
    def __init__(self, s_t, a_t, r_t):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = r_t
def log_init():
    if os.path.exists('loss.txt'):
        os.remove('loss.txt')
def log_train(loss, param_norm = None):
    with open('loss.txt', 'a') as writer:
        writer.write(str(loss) + '\n')
def log_eval(reward):
    print('Eval reward', reward)
    
def evaluate(env, model, img_size, steps=1000, visualize=False):
    prev_s_t = None
    rewards = []
    action = env.action_space.sample()
    for i in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)

        # Resize frame to desired size, e.g., 320x240
        frame = cv2.resize(observation, (img_size, img_size), interpolation=cv2.INTER_AREA)
        action = model.sample_action(frame)
        
        if visualize:
            cv2.imshow("Custom Render", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        rewards.append(reward)

        if terminated or truncated:
            observation, info = env.reset()
    return np.mean(rewards)