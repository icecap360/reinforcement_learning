import gymnasium as gym
valid_environments = ['Breakout', 'Enduro', 'Pong', 'SpaceInvaders', 'Seaquest', 'BeamRider' ]
env = gym.make("ALE/Pong")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()