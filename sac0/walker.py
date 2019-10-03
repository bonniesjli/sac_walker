import argparse
import datetime
from box import Box
from gym import spaces
from mlagents.envs import UnityEnvironment
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="walker",
                    help='System launching Unity ML Agents')
parser.add_argument('--system', default="window",
                    help='System launching Unity ML Agents')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
from fetch_env import fetch_env
env = UnityEnvironment(fetch_env(args.env, args.system))
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

from pprint import pprint
pprint(env)
pprint(brain)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
num_worker = 11
state_dim = 1060
high = np.ones(39)
action_dim = spaces.Box(-high, high, dtype=np.float32)
agent = SAC(state_dim, action_dim, args)

#TesnorboardX
writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Training Loop
total_numsteps = 0
updates = 0

agent_reward = np.zeros(num_worker)
buffer_reward = np.zeros(num_worker)
episode_steps = 0
done = False
env_info = env.reset(train_mode = True)[default_brain]
states = env_info.vector_observations

while total_numsteps <= args.num_steps:

    actions = agent.select_action(states)  # Sample action from policy

    # if len(memory) > args.batch_size:
    #     # Number of updates per step in environment
    #     for i in range(args.updates_per_step):
    #         # Update parameters of all the networks
    #         critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
    #
    #         writer.add_scalar('loss/critic_1', critic_1_loss, updates)
    #         writer.add_scalar('loss/critic_2', critic_2_loss, updates)
    #         writer.add_scalar('loss/policy', policy_loss, updates)
    #         writer.add_scalar('loss/entropy_loss', ent_loss, updates)
    #         writer.add_scalar('entropy_temprature/alpha', alpha, updates)
    #         updates += 1

    env_info = env.step(actions)[default_brain] # Step
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    transition = (states, actions, rewards, next_states, dones)
    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.step(transition)

    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
    writer.add_scalar('loss/policy', policy_loss, updates)
    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
    writer.add_scalar('entropy_temprature/alpha', alpha, updates)

    total_numsteps += 1
    agent_reward += rewards
    for j, d in enumerate(dones):
        if dones[j]:
            buffer_reward[j] = agent_reward[j]
            agent_reward[j] = 0

    # Ignore the "done" signal if it comes from hitting the time horizon.
    # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
    # mask = 1 if episode_steps == env._max_episode_steps else float(not done)

    states = next_states

    if total_numsteps % 20 == 0:
        writer.add_scalar('reward/train', np.mean(buffer_reward), total_numsteps)
        print("total numsteps: {}, reward: {}".format(total_numsteps, np.mean(buffer_reward)))

    if total_numsteps % 100 == 0 and args.eval == True:
        agent_reward = np.zeros(num_worker)
        buffer_reward = np.zeros(num_worker)
        env_info = env.reset()[default_brain]
        states = env_info.vector_observations
        d = 0
        while d != num_worker:
            actions = agent.select_action(states, eval=True)
            env_info = env.step(actions)[default_brain]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent_reward += rewards
            for j, d in enumerate(dones):
                if dones[j]:
                    buffer_reward[j] = agent_reward[j]
                    agent_reward[j] = 0
            d = 0
            for r in buffer_reward:
                if r != 0:
                    d += 1
            states = next_states

        writer.add_scalar('avg_reward/test', np.mean(buffer_reward))

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(total_numsteps, np.mean(buffer_reward)))
        print("----------------------------------------")
        agent_reward = np.zeros(num_worker)
        buffer_reward = np.zeros(num_worker)
        env_info = env.reset()[default_brain]
        states = env_info.vector_observations

env.close()
