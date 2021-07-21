"""model.py: Implementation of methods to understand statistical inter-relations between the variables of interest
in the generative model"""

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
            factor = ((sigma_v ** 2) * (sigma_a ** 2)) + ((sigma_v ** 2) * (sigma_p ** 2)) \
                     + ((sigma_p ** 2) * (sigma_a ** 2))
            exponent = ((((x_v - x_a) ** 2) * (sigma_p ** 2)) + (((x_v - mu_p) ** 2) * (sigma_a ** 2)) +
                        (((x_a - mu_p) ** 2) * (sigma_v ** 2))) / factor
            probability = np.exp(-exponent / 2) / (np.sqrt(factor) * 2 * np.pi)
        # p(x_v, x_a | C = 2) ~ eq. 6
        elif cause == 'separate':
            factor = ((sigma_v ** 2) + (sigma_p ** 2)) * ((sigma_a ** 2) + (sigma_p ** 2))
            exponent = (((x_v - mu_p) ** 2) / ((sigma_v ** 2) + (sigma_p ** 2))) + (((x_a - mu_p) ** 2) /
                                                                                    ((sigma_a ** 2) + (sigma_p ** 2)))
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
            probability_common = self.probability_cause(x_v, x_a, cause='common', p_common=p_common, sigma_v=sigma_v,
                                                        sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
            probability_separate = self.probability_cause(x_v, x_a, cause='separate', p_common=p_common,
                                                          sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p, mu_p=mu_p)
            c_1 = probability_common * self.estimate_best_signal(x_v=x_v, x_a=x_a, cause='common', type=type,
                                                                 sigma_v=sigma_v, sigma_a=sigma_a, sigma_p=sigma_p,
                                                                 mu_p=mu_p)
            c_2 = probability_separate * self.estimate_best_signal(x_v=x_v, x_a=x_a, cause='separate', type=type,
                                                                   sigma_v=sigma_v, sigma_p=sigma_p, mu_p=mu_p)
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

        probability_common = self.probability_signal(x_v, x_a, cause='common', sigma_v=sigma_v, sigma_a=sigma_a,
                                                     sigma_p=sigma_p, mu_p=mu_p)
        probability_separate = self.probability_signal(x_v, x_a, cause='separate', sigma_v=sigma_v, sigma_a=sigma_a,
                                                       sigma_p=sigma_p, mu_p=mu_p)
        # p(C = 1 | x_v, x_a)  ~ eq. 2
        probability_common_cause = (probability_common * p_common) / (
                (probability_common * p_common) + (probability_separate * (1 - p_common)))

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
        cause = np.random.binomial(1, probability_common, n) + 1
        is_common = np.logical_not((cause - 1).astype(bool))

        stimulus_pairs[cause == 1, 0] = stimulus_pairs[cause == 1, 1] = np.random.normal(0, sigma_p, np.sum(cause == 1))
        stimulus_pairs[cause == 2, :] = np.random.normal(0, sigma_p, (np.sum(cause == 2), 2))
        
        # map stimulus pairs
        stimulus_pairs[:, 0] = [min(self.s_v, key=lambda x:abs(x-i)) for i in stimulus_pairs[:, 0]]
        stimulus_pairs[:, 1] = [min(self.s_a, key=lambda x:abs(x-i)) for i in stimulus_pairs[:, 1]]

        return stimulus_pairs, is_common

    def make_button_presses(self, stimulus_pairs, bins=None, plot=True, **kwargs):
        # TODO: docstring
        # stimulus_pairs = np.array of [s_v, s_a]'s

        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
        bins = self.range if bins is None else bins

        histogram_vs, histogram_as = [], []
        stimulus_pairs_unique, count = np.unique(stimulus_pairs, axis=0, return_counts=True)
        
        for (s_v, s_a), n in zip(stimulus_pairs_unique, count):
            x_v = np.random.normal(s_v, sigma_v, n)
            x_a = np.random.normal(s_a, sigma_a, n)
            s_v_estimate = self.estimate_signal(x_v, x_a, 'video')
            s_a_estimate = self.estimate_signal(x_v, x_a, 'audio')

            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            histogram_v, _ = np.histogram(s_v_estimate, bins)/n
            histogram_a, _ = np.histogram(s_a_estimate, bins)/n
            histogram_vs.append(histogram_v)
            histogram_as.append(histogram_a)     

        histogram_vs, histogram_as = np.array(histogram_vs), np.array(histogram_as)
        
        # TODO: shift plotting to plot.py
        if plot:
            fig, axs = plt.subplots(len(self.s_v), len(self.s_a), figsize=(20, 20))
            for (i, j), k in zip(np.array(list(product(range(len(self.s_v)), range(len(self.s_a))))), range(len(stimulus_pairs_unique))):
            # for i in range(len(self.s_v)):
            #     for j in range(len(self.s_a)):
                axs[i][j].bar(self.s_v, histogram_vs[k], tick_label=self.s_v, alpha=.7, label='video')
                axs[i][j].bar(self.s_a, histogram_as[k], tick_label=self.s_a, alpha=.7, label='audio')
                axs[i][j].set_xlabel('Position estimates; $\hat{s_V}$, $\hat{s_V}$')
                axs[i][j].set_ylabel('Probability')
                axs[i][j].set_title('Position estimates for $s_V$=%.1f, $s_a$=%.1f,' % (stimulus_pairs_unique[k][0], stimulus_pairs_unique[k][1]))
                
            fig.tight_layout()
            plt.legend(loc="upper left", bbox_to_anchor=(1,0))
            plt.show()
               
        return np.array(histogram_vs), np.array(histogram_as)

    # Log likelihood calculation (1d)
    def log_likelihood(self, trials=10000, eps=1e-5):
        s_hat_v_hist, s_hat_a_hist = self.make_button_presses(trials, plot=False)
        n_v = np.sum(s_hat_v_hist,axis=0)/np.sum(s_hat_v_hist)  # observed response counts, per auditory condition
        n_a = np.sum(s_hat_a_hist,axis=0)/np.sum(s_hat_a_hist)  # ... and visual condition

        s_hat_v_hist_model, s_hat_a_hist_model = self.make_button_presses(trials * 10, plot=False)

        p_a = np.sum(s_hat_a_hist_model, axis = 0) / np.sum(s_hat_a_hist_model)  # probability of button press for each auditory bin
        p_v = np.sum(s_hat_v_hist_model, axis = 0) / np.sum(s_hat_v_hist_model)

        log_like = np.zeros(2)  # one for s_v, one for s_a

        log_like[0] = (n_a * np.log(p_a + eps)).sum()  # log likelihood for sigma_a
        log_like[1] = (n_v * np.log(p_v + eps)).sum()  # log likelihood for sigma_v

        # TODO: do not know whether the two log likelihoods should be separated or we should have
        # joined them somehow in a previous step.
        return log_like.sum()/2  # returns log likelihood of parameters fitting data

    # Brute fitting (1f)
    def brute_fitting(self, n_sample=10, trials = 10000):
        """Performing of the fitting to see which parameter combination is best

        """

        p_common = np.linspace(0, 1, n_sample)
        sigmas_v = sigmas_a = sigmas_p = np.linspace(0.1, 20, n_sample)
        likelihood = np.zeros((int(n_sample ** 4)))
        parameters = np.zeros((int(n_sample ** 4), 4))
        num = 0

        for p in p_common:
            self.p_common = p
            for sv in sigmas_v:
                self.sigma_p = sv
                for sa in sigmas_a:
                    self.sigma_a = sa
                    for sp in sigmas_p:
                        self.sigma_p = sp
                        likelihood[num] = self.log_likelihood(trials)
                        parameters[num, :] = np.array([p, sv, sa, sp])
                        print(likelihood[num])
                        num += 1

        return likelihood, parameters

    # Section g)
    def log_posterior(self, parameter1, parameter2, num_bins=10, eps=1e-10):

        parameters = np.linspace(parameter1, parameter2, num_bins)
        log_like = np.zeros(num_bins)
        for i, param in enumerate(parameters):
            ## TODO: State which parameter needs to be changed.
            # self.x = param
            log_like[i] = self.log_likelihood()

        prior = rectangular_prior(parameter1, parameter2, num_bins)[0]
        log_prior = np.log(prior/np.sum(prior))
        log_post = log_like + log_prior

        return log_post


def rectangular_prior(parameter1, parameter2, num_bins=10, n_points=10000):
    """ Creating the rectangular prior.
    """
    values = np.random.uniform(parameter1, parameter2, n_points)
    distribution = np.histogram(values, num_bins-1)

    return distribution
