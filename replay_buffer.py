"""
ReplayBuffer class file.
"""

from collections import deque
import random


class ReplayBuffer():
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data = state, action, return, advant.
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of state, action, return, advant.
        """
        self.buffer.append(data)

    def pull(self, data_size):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :return:
            A tuple which consists of lists of state, action, return, advant.
        """
        minibatch = random.sample(self.buffer, data_size)
        s_lst, a_lst, r_lst, ad_lst, op_lst = [], [], [], [], []

        for state, action, advant, a_return, old_prob in minibatch:
            s_lst.append(state)
            a_lst.append(action)
            ad_lst.append(advant)
            r_lst.append(a_return)
            op_lst.append(old_prob)

        return s_lst, a_lst, ad_lst, r_lst, op_lst

    def size(self):
        """
        Returns the size of buffer.
        """
        return len(self.buffer)
