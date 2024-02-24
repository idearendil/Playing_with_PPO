"""
The file of ObservationNormalizer class
which makes an observation much easier for networks.
"""
import numpy as np


class ObservationNormalizer:
    """
    When an observation gets in, normalize it
    with mean and standard deviation of previous observations.
    """
    def __init__(self, s_dim):
        self.dim = s_dim
        self.mean = np.zeros((self.dim,))
        self.std = np.zeros((self.dim, ))
        self.stdd = np.zeros((self.dim, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean
        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)

        return x

    def save(self, path):
        """
        Save current ObservationNormalizer state.
        """
        single_np = np.zeros((1,))
        single_np[0] = self.n
        total_np = np.concatenate(
            (self.mean, self.std, self.stdd, single_np), axis=0)
        np.save(path + 'observation_normalizer', total_np)

    def load(self, path):
        """
        Load previous ObservationNormalizer state.
        """
        total_np = np.load(path + 'observation_normalizer.npy')
        self.mean = total_np[:self.dim]
        self.std = total_np[self.dim:self.dim*2]
        self.stdd = total_np[self.dim*2:self.dim*3]
        self.n = total_np[-1]
