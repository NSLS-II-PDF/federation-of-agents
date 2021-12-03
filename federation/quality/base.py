from typing import Callable
import numpy as np
from collections import Counter


class Agent:
    def __init__(
        self, n_samples: int, quality_function: Callable[[np.array], int] = None
    ):
        """
        Agent class that retains a counter of measurements at each sample,
        the index of the current sample, and a quality array with the current sample quality
        of each sample.

        Quality should be given as a natural number starting from 1 to the trained maximum.
        A regular default is to use {1: bad, 2: mediocre, 3:good}.
        It is expected that the sample quality can and should improve over time, and will be
        updated in the `tell` method as provided by the document stream.

        Parameters
        ----------
        n_samples: int
            Number of samples in measurement
        """
        self.counter = Counter()  # Counter of measurements
        self.current = None  # Current sample
        self.n_samples = n_samples
        self.cum_sum = dict()
        self.quality = np.zeros(self.n_samples)  # Current understood quality
        if quality_function is None:
            self.quality_function = self._default_quality
        else:
            self.quality_function = quality_function

    @staticmethod
    def _default_quality(arr) -> int:
        """Uses a proxy for Signal to Noise to break into 3 tiers."""
        SNR = np.max(arr) / np.mean(arr)
        if SNR < 2:
            return 1
        elif SNR < 3:
            return 2
        else:
            return 3

    def tell(self, x=None, y=None):
        """
        Tell's based on current sample only
        Parameters
        ----------
        x: float, array
        y: float, array

        Returns
        -------

        """
        if x is not None:
            self.current = x
        self.counter[self.current] += 1
        if self.current in self.cum_sum:
            self.cum_sum[self.current] += y
        else:
            self.cum_sum[self.current] = y
        self.quality[self.current] = self.quality_function(self.cum_sum[self.current])

    def tell_many(self, xs, ys):
        """Useful for reload"""
        for x, y in zip(xs, ys):
            self.counter[x] += 1
            if self.current in self.cum_sum:
                self.cum_sum[x] += y
            else:
                self.cum_sum[x] = y
        for i in range(self.n_samples):
            self.quality[i] = self.quality_function(self.cum_sum[i])

    def ask(self, n):
        raise NotImplementedError


class SequentialAgent(Agent):
    def __init__(self, n_samples):
        """
        Sequential agent that just keeps on going.

        Agent parent class retains a counter of measurements at each sample,
        the index of the current sample, and a quality array with the current sample quality
        of each sample.

        Quality should be given as a natural number starting from 1 to the trained maximum.
        A regular default is to use {1: bad, 2: mediocre, 3:good}.
        It is expected that the sample quality can and should improve over time, and will be
        updated in the `tell` method as provided by the document stream.

        Parameters
        ----------
        n_samples: int
            Number of samples in measurement
        """
        super().__init__(n_samples)

    def ask(self, n):
        return (self.current + 1) % self.n_samples


class MarkovAgent(Agent):
    def __init__(self, n_samples, max_quality, min_quality=1, seed=None):
        """
        Stochastic agent that moves preferentially to worse seeds.
        Queries a random transition and accepts with a probability of badness divided by range of quality.

        Agent parent class retains a counter of measurements at each sample,
        the index of the current sample, and a quality array with the current sample quality
        of each sample.

        Quality should be given as a natural number starting from 1 to the trained maximum.
        A regular default is to use {1: bad, 2: mediocre, 3:good}.
        It is expected that the sample quality can and should improve over time, and will be
        updated in the `tell` method as provided by the document stream.

        Parameters
        ----------
        n_samples: int
            Number of samples in measurement
        max_quality: int
            Maximum quality value
        min_quality: int
            Minimum quality value. Should be 1 unless you're doing something strange.
        """
        super().__init__(n_samples)
        self.max_quality = max_quality
        self.min_quality = min_quality
        self.rng = np.random.default_rng(seed)

    def ask(self, n):
        accept = False
        proposal = None
        while not accept:
            proposal = self.rng.integers(self.n_samples)
            if self.rng.random() < (self.max_quality - self.quality[proposal]) / (
                self.max_quality - self.min_quality
            ):
                accept = True

        return proposal
