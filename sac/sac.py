import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from model import ValueNetwork, SoftQNetwork, PolicyNetwork
from utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC():
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 buffer_size = int(1e6),
                 batch_size = 128,
                 learning_rate = 3e-4,
                 learn_every = 4):

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion = nn.MSELoss()
        self.value_criterion = nn.MSELoss()
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.t_step = 0
        self.learn_every = learn_every
        self.batch_size = batch_size

    def act(self, state):
        action = self.policy_net.get_action(state)
        return action

    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add(state, action, reward, next_state, done)
        self.t_step += 0
        if self.t_step % self.learn_every == 0:
            if self.buffer.__len__() > self.batch_size:
                batch = self.buffer.sample()
                self.learn(batch)

    def learn(self,
              batch,
              gamma=0.99,
              mean_lambda=1e-3,
              std_lambda=1e-3,
              z_lambda=0.0):

        state, action, reward, next_state, done = batch

        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_update()
        # print ("learning iter complete")

    def soft_update(self, soft_tau=1e-2):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
