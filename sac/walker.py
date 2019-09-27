from sac import SAC
from utils import *
from loggin import *
from torch.multiprocessing import Pipe
from mlagents.envs import UnityEnvironment
import numpy as np
import time
import sys

print("Python version:")
print(sys.version)
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")
arg1 = sys.argv[1]
if arg1 == "linux":
    env = UnityEnvironment(file_name = "../envs/walker_linux/pyramid.x86_64")
if arg1 == "window":
    env = UnityEnvironment(file_name = "../envs/walker_window/Unity Environment.exe")

default_brain = env.brain_names[0]
brain = env.brains[default_brain]

def main(run):
    """
    :param: (str) run
    """
    env.reset()

    max_t = 1.5e6
    t_horizon = 10
    t = 0
    num_worker = 11
    state_dim = 1060
    action_dim = 39
    agent = SAC(state_dim, action_dim, hidden_dim = 512)

    pre_obs_norm_step = 10000
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(1, state_dim)
    steps = 0
    next_obs = []
    print('Start to initialize observation normalization ...')
    while steps < pre_obs_norm_step:
        steps += num_worker
        actions = [np.random.randn(action_dim) for _ in range(num_worker)]
        env_info = env.step(actions)[default_brain]
        obs = env_info.vector_observations
        for o in obs:
            next_obs.append(o)
    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initialize')

    LOG = Logging(run)
    LOG.create("score")
    LOG.create("full_score")

    agent_r = np.zeros(num_worker)
    buffer_r = np.zeros(num_worker)
    env_info = env.reset(train_mode=True)[default_brain]
    states = env_info.vector_observations
    while t <= max_t:
        t += 1
        action = agent.act((np.float32(states) - obs_rms.mean)/np.sqrt(obs_rms.var))
        env_info = env.step(actions)[default_brain]

        obs = env_info.vector_observations
        next_states = obs
        rewards = env_info.rewards
        dones = env_info.local_done

        agent_r += rewards
        for j, d in enumerate(dones):
                if dones[j]:
                    buffer_r[j] = agent_r[j]
                    agent_r[j] = 0

        agent.step((np.float32(states) - obs_rms.mean)/np.sqrt(obs_rms.var),
                    actions,
                    rewards,
                    (np.float32(next_states) - obs_rms.mean) / np.sqrt(obs_rms.var),
                    dones)

        states = next_states
        if t % 1000 == 0:
            LOG.log("score", np.mean(buffer_r))

        if t < 10000 and t % 2000 == 0:
            print('\rTimeStep {}\tAverage Score: {:.2f}'.format(t, LOG.mean("score", t_horizon)))
            LOG.save_data()
        if t % 10000 == 0:
            print('\rTimeStep {}\tAverage Score: {:.2f}'.format(t, LOG.mean("score", t_horizon)))
            torch.save(agent.policy_net, 'policy.pt')
            LOG.save_data()

    LOG.save_data()
    LOG.visualize("score")

if __name__ == '__main__':
    main("run1")
