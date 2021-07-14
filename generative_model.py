"""model.py: Implementation of methods to understand statistical inter-relations between the variables of interest
in the generative model"""

# from typing import Optional
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
class GenerativeModel:

    def __init__(self):
        self.s_v = np.array([-12, -6, 0, 6, 12])
        self.s_a = np.array([-12, -6, 0, 6, 12])
        self.p_common = 0.8
        self.sigma_v = 0.6
        self.sigma_a = 3.1
        self.sigma_p = 15
        self.mu_p = 0
        self.range = np.array([-np.inf, -9, -3, 3, 9, np.inf])

    def probability_signal(self, x_v, x_a, cause='', **kwargs):
        """
        Estimates the probability of the audio and visual signals(noisy) given the cause.

        Parameters
        ----------
        x_v: np.array
            array of the current board state
        x_a: np.array
            player making the move
        cause: str
            saved state of the players
        #TODO: add all parameters in docstring

        Returns
        -------
        Probability of the audio and visual signals(noisy) for the given cause.
        """
        # TODO: common method for initialisation
        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)
        mu_p = kwargs.get('mu_p', self.mu_p)

        # p(x_v, x_a | C = 1) ~ eq. 4
        if cause == 'common':
            factor = ((sigma_v ** 2) * (sigma_a ** 2)) + ((sigma_v ** 2) * (sigma_p ** 2)) + ((sigma_p ** 2) * (sigma_a ** 2))
            exponent = ((((x_v - x_a) ** 2) * (sigma_p ** 2)) + (((x_v - mu_p) ** 2) * (sigma_a ** 2)) + (((x_a - mu_p) ** 2) * (sigma_v ** 2))) / factor
            probability = np.exp(-exponent / 2) / (np.sqrt(factor) * 2 * np.pi)
        # p(x_v, x_a | C = 2) ~ eq. 6
        elif cause == 'separate':
            factor = ((sigma_v ** 2) + (sigma_p ** 2)) * ((sigma_a ** 2) + (sigma_p ** 2))
            exponent = (((x_v - mu_p) ** 2) / ((sigma_v ** 2) + (sigma_p ** 2))) + (((x_a - mu_p) ** 2) / ((sigma_a ** 2) + (sigma_p ** 2)))
            probability = np.exp(-exponent / 2) / (np.sqrt(factor) * 2 * np.pi)
        else:
            raise ValueError("Cause must either be 'common' or 'separate'!")

        return probability

    def estimate_signal(self, x_v, x_a, type='', **kwargs):
        # TODO: docstring

        p_common = kwargs.get('p_common', self.p_common)
        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)
        mu_p = kwargs.get('mu_p', self.mu_p)

        # s_hat_v ~ eq. 9
        if type == 'audio' or type == 'video':
            probability_common = self.probability_cause(x_v, x_a, cause='common', p_common=p_common, sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
            probability_separate = self.probability_cause(x_v, x_a, cause='separate', p_common=p_common, sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
            c_1 = probability_common * self.estimate_best_signal(x_v=x_v, x_a=x_a, cause='common', type=type, sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
            c_2 = probability_separate * self.estimate_best_signal(x_v=x_v, x_a=x_a, cause='separate', type=type, sigma_v=sigma_v, sigma_p=sigma_p, mu_p=mu_p)
        else:
            raise ValueError("Type must either be 'video' or 'audio'!")

        return c_1 + c_2

    def estimate_best_signal(self, x_v=0, x_a=0, cause='', type='', **kwargs):
        # TODO: docstring

        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)
        mu_p = kwargs.get('mu_p', self.mu_p)

        # eq. 12
        if cause == 'common':
            dividend = x_v / (sigma_v ** 2) + x_a / (sigma_a ** 2) + mu_p / (sigma_p ** 2)
            divisor = (1 / (sigma_v ** 2)) + (1 / (sigma_a ** 2)) + (1 / (sigma_p ** 2))

        # eq. 11
        elif cause == 'separate':
            if type == 'video':
                dividend = x_v / (sigma_v ** 2) + mu_p / (sigma_p ** 2)
                divisor = (1 / (sigma_v ** 2)) + (1 / (sigma_p ** 2))
            elif type == 'audio':
                dividend = x_a / (sigma_a ** 2) + mu_p / (sigma_p ** 2)
                divisor = (1 / (sigma_a ** 2)) + (1 / (sigma_p ** 2))
            else:
                raise ValueError("Type must either be 'video' or 'audio'!")

        else:
            raise ValueError("Cause must either be 'common' or 'separate'!")

        return dividend / divisor

    def probability_cause(self, x_v, x_a, cause='common', **kwargs):
        # TODO: docstring

        p_common = kwargs.get('p_common', self.p_common)
        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)
        mu_p = kwargs.get('mu_p', self.mu_p)

        probability_common = self.probability_signal(x_v, x_a, cause='common', sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
        probability_separate = self.probability_signal(x_v, x_a, cause='separate', sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
        # p(C = 1 | x_v, x_a)  ~ eq. 2
        probability_common_cause = (probability_common * p_common) / ((probability_common * p_common) + (probability_separate * (1 - p_common)))

        if cause == 'common':
            return probability_common_cause
        elif cause == 'separate':
            return 1 - probability_common_cause
        else:
            raise ValueError("Cause must either be 'common' or 'separate'!")

    def generate_stimulus_pairs(self, n, **kwargs):
        """
        Generate n stimulus pairs.

        Parameters
        ----------
        n: int
            number of stimulus pairs to be generated

        Returns
        ----------
        stimulus_pairs: np.array
            array with the s values of the pairs
        is_common: bool
            True if index of stimulus pair has a common cause
        """
        sigma_p = kwargs.get('sigma_p', self.sigma_p)

        stimulus_pairs = np.zeros((n, 2))
        probability_common = np.random.rand(n)
        cause = np.random.binomial(1, probability_common, n)+1
        is_common = np.logical_not((cause-1).astype(bool))

        stimulus_pairs[cause==1, 0] = stimulus_pairs[cause==1, 1] = np.random.normal(0, sigma_p, np.sum(cause==1))
        stimulus_pairs[cause==2, :] = np.random.normal(0, sigma_p, (np.sum(cause==2), 2))

        return stimulus_pairs, is_common

    def make_button_presses(self, n, stimulus_pairs=None, plot=True, **kwargs):
        # TODO: docstring
        #stimulus_pairs = np.array of [s_v, s_a]'s

        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)

        if stimulus_pairs is None:
            stimulus_pairs = np.array(list(product(self.s_v, self.s_a)))
        
        v_histogram = a_histogram = []

        for s_v, s_a in stimulus_pairs:
            x_v = np.random.normal(s_v, sigma_v, n)
            x_a = np.random.normal(s_a, sigma_a, n)
            s_v_estimate = self.estimate_signal(x_v, x_a, 'video')
            s_a_estimate = self.estimate_signal(x_v, x_a, 'audio')
            
            histogram_v, _ = np.histogram(s_v_estimate, self.range)
            histogram_a, _ = np.histogram(s_a_estimate, self.range)

            v_histogram.append(histogram_v)
            a_histogram.append(histogram_a)
            
            if plot:
                x = np.sort(np.unique(stimulus_pairs[:, 0]))
                plt.bar(x, histogram_v, tick_label=x, alpha=.7, label='video')
                x = np.sort(np.unique(stimulus_pairs[:, 1]))
                plt.bar(x, histogram_a, tick_label=x, alpha=.7, label='audio')
                plt.xlabel('Position estimates; $\hat{s_V}$, $\hat{s_V}$')
                plt.ylabel('Count')
                plt.title('Position estimates for $s_V$=%.1f, $s_a$=%.1f,'%(s_v, s_a))
                plt.legend()
                plt.show()

        return v_histogram, a_histogram
