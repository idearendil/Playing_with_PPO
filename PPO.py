"""
The file of Ppo class.
"""

import torch.optim as optim
import numpy as np
import torch
from model import Actor, Critic
from parameters import LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, BATCH_SIZE, \
    EPSILON, L2_RATE, BUFFER_SIZE, BATCH_NUM
from replay_buffer import ReplayBuffer


class Ppo:
    """
    The class which Proximal Policy Optimization is implemented in.
    """
    def __init__(self, s_dim, a_dim, device):
        self.device = device
        self.actor_net = Actor(s_dim, a_dim).to(self.device)
        self.critic_net = Critic(s_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=L2_RATE
        )
        self.critic_loss_func = torch.nn.MSELoss()
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def models_to_device(self, device):
        """
        Move actor and critic to the specific device('cpu' or 'cuda:x')
        """
        self.actor_net.to(device)
        self.critic_net.to(device)

    def push_an_episode(self, data):
        """
        Push an episode to replay buffer.

        Args:
            data: an array of (state, action, reward, mask).
        Returns:
            None
        """

        state_lst, action_lst, reward_lst, mask_lst, prob_lst = \
            [], [], [], [], []
        for a_state, a_action, a_reward, a_mask, a_prob in data:
            state_lst.append(a_state)
            action_lst.append(a_action)
            reward_lst.append(a_reward)
            mask_lst.append(a_mask)
            prob_lst.append(a_prob)

        states = torch.Tensor(
            np.array(state_lst, dtype=np.float32)).to(self.device)
        actions = torch.Tensor(
            np.array(action_lst, dtype=np.float32)).to(self.device)
        rewards = torch.Tensor(np.array(reward_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))
        old_probs = torch.Tensor(np.array(prob_lst, dtype=np.float32))

        with torch.no_grad():
            self.critic_net.eval()
            values = self.critic_net(states)
            returns, advants = self.get_gae(rewards, masks, values.cpu())

        for idx, _ in enumerate(states):
            self.buffer.push((states[idx],
                              actions[idx],
                              advants[idx],
                              returns[idx],
                              old_probs[idx]))

    def train(self):
        """
        Train Actor network and Value network with data in buffer.
        """

        self.actor_net.train()
        self.critic_net.train()
        for _ in range(BATCH_NUM):

            states, actions, advants, returns, old_probs = self.buffer.pull(
                BATCH_SIZE)
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            advants = torch.stack(advants).unsqueeze(1).to(self.device)
            returns = torch.stack(returns).unsqueeze(1).to(self.device)
            old_probs = torch.stack(old_probs).to(self.device)

            values = self.critic_net(states)
            critic_loss = self.critic_loss_func(values, returns)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            mu, std = self.actor_net(states)
            pi = self.actor_net.distribution(mu, std)
            new_prob = pi.log_prob(actions).sum(1, keepdim=True)

            ratio = torch.exp(new_prob - old_probs)
            surrogate_loss = ratio * advants

            ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = ratio * advants

            actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

    def kl_divergence(self, old_mu, old_sigma, mu, sigma):
        """
        KL divergence of two different normal distributions.
        """
        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()
        kl = (
            torch.log(old_sigma)
            - torch.log(sigma)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5
        )
        return kl.sum(1, keepdim=True)

    def get_gae(self, rewards, masks, values):
        """
        Calculate Generalized Advantage Estimation.
        """
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + GAMMA * running_returns * masks[t]
            running_tderror = (
                rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
            )
            running_advants = (
                running_tderror + GAMMA * LAMBDA * running_advants * masks[t]
            )

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants
