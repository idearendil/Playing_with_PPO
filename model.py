"""
The file of actor and critic architectures.
"""
import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.sigma = nn.Linear(256, a_dim)
        self.mu = nn.Linear(256, a_dim)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal

    def set_init(self, layers):
        """
        Weight initialization method(but not used now)
        """
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            mu and sigma of policy considering current observation
        """
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def choose_action(self, s):
        """
        Choose action by normal distribution

        Args:
            s: current observation
        Returns:
            action tensor sampled from policy(normal distribution),
            log probability of the action
        """
        mu, sigma = self.forward(s)
        pi = self.distribution(mu, sigma)
        an_action = pi.sample().squeeze()
        action_prob = pi.log_prob(an_action).sum(1, keepdim=True).squeeze(0)
        return an_action.cpu().numpy(), action_prob.cpu()


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, N_S):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_S, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        """
        Weight initialization method(but not used now)
        """
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            estimated value of current state
        """
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values
