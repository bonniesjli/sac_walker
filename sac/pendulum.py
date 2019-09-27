from sac import SAC
from utils.pendulum import NormalizedActions, plot
import numpy as np

import gym
env = NormalizedActions(gym.make("Pendulum-v0"))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

agent = SAC(state_dim, action_dim, hidden_dim = 32)

max_frames  = 20000
max_steps   = 500
frame_idx   = 0

from loggin import *
LOG = Logging("pendulum")
LOG.create("score")

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = np.zeros(1)

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = np.expand_dims(reward, 1)
        done = np.array([1 if done else 0])
        agent.step(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if frame_idx % 1000 == 0:
            plot(frame_idx, rewards)
            print("10 episode avg reward:", LOG.mean("score", 10))

        if done:
            break
    LOG.log("score", episode_reward, frame_idx)

LOG.save_data()
LOG.visualize("score")
