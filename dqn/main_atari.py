import gymnasium as gym
import cv2
from model import QFunctionConv, MainModel
import torch.nn as nn
import torch
import random 
import numpy as np
import os
from misc import log_train, log_init, Replay, evaluate, log_eval
# import ale_py
# # if using gymnasium
# import shimmy
# import gym # or "import gymnasium as gym"
# print(gym.envs.registry.keys())

valid_environments = ['Breakout', 'Pong', 'SpaceInvaders',  'Enduro', 'Seaquest', 'BeamRider' ]
env = gym.make("ALE/Pong", render_mode="rgb_array")
observation, info = env.reset(seed=42)
visualize = False
print_reward = False
device = 'cuda:0'
img_size = 256
n_train_iters = int(1e6)
n_start_train_iters = 1000
p_eval = 1e3
batch_size = 64

assert batch_size < n_start_train_iters
n_actions = env.action_space.n
replay_buffer = []

model = MainModel(
    QFunctionConv(
        img_size,
        env.action_space.n
    ), 
    device=device)

model.to(device)
optimizer = torch.optim.AdamW(model.q_function.parameters(), lr=1e-4)
log_init()
prev_s_t = None

for i in range(n_train_iters):
    epsilon = min(1, n_start_train_iters/(i+1) + 0.05)
    rand_float = random.random()
    if rand_float > epsilon:
        action = model.sample_action(prev_s_t)
    else:
        action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Resize frame to desired size, e.g., 320x240
    frame = cv2.resize(observation, (img_size, img_size), interpolation=cv2.INTER_AREA)
    prev_s_t = frame

    replay_buffer.append(Replay(frame, action, reward))

    if i > n_start_train_iters:
        optimizer.zero_grad()
        batch = random.sample(replay_buffer, batch_size)
        loss = model.calculate_loss(batch)
        loss.backward()
        optimizer.step()
        log_train(loss)

    if i % p_eval == 0:
        reward = evaluate(env, model, img_size, visualize=visualize)
        log_eval(reward)
    
    if visualize:
        cv2.imshow("Custom Render", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if terminated or truncated:
        observation, info = env.reset()
env.close()
cv2.destroyAllWindows()