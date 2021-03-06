"""model.py: Implementation of methods to understand statistical inter-relations between the variables of interest
in the generative model"""

from __future__ import division
from plot import plot_button_press_histograms
import emcee
import numpy as np
import matplotlib.pyplot as plt

class GenerativeModel:

    def __init__(self):
        """Initialises model parameters.
        """
        self.s_v = np.array([-12, -6, 0, 6, 12])
        self.s_a = np.array([-12, -6, 0, 6, 12])
        self.p_common = 0.8
        self.sigma_v = 0.6
        self.sigma_a = 3.1
        self.sigma_p = 15
        self.mu_p = 0
        self.range = np.array([-np.inf, -9, -3, 3, 9, np.inf])

    def probability_signal(self, x_v, x_a, cause='', **kwargs):
        """Estimates the probability of the audio and visual signals(noisy) given the cause.

        Args:
            x_v (np.array): noisy video signal
            x_a (np.array): noisy audio signal
            cause (str, optional): cause type of the signal - common or separate. Defaults to ''.

        Raises:
            ValueError: If cause is not 'common' or 'separate'.

        Returns:
            float: probability
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
        """Estimates the signal given the noise audio and video signal.

        Args:
            x_v (np.array): noisy video signal
            x_a (np.array): noisy audio signal
            type (str, optional): Type of signal - audio or video. Defaults to ''.

        Raises:
            ValueError: If type is not 'audio' or 'video'.

        Returns:
            float: estimated signal
        """
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
        """Generates n stimulus pairs.

        Args:
            n (int): number of stimulus pairs to be generated

        Returns:
            np.array: array with the s values of the pairs
        """
        p_common = kwargs.get('p_common', self.p_common)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)

        stimulus_pairs = np.zeros((n, 2))
        cause = np.random.binomial(1, p_common, n) + 1

        stimulus_pairs[cause == 1, 0] = stimulus_pairs[cause == 1, 1] = np.random.normal(0, sigma_p, np.sum(cause == 1))
        stimulus_pairs[cause == 2, :] = np.random.normal(0, sigma_p, (np.sum(cause == 2), 2))
        
        # map stimulus pairs
        s_v_expanded = np.ones((stimulus_pairs.shape[0],len(self.s_v), stimulus_pairs.shape[1]))*np.array((self.s_v, self.s_a)).T
        stimulus_pairs = self.s_v[np.argmin(np.abs(s_v_expanded - np.expand_dims(stimulus_pairs, axis=1)), axis=1)]

        return stimulus_pairs
    
    def make_button_presses(self, stimulus_pairs, bins=None, plot=True, saved_image=False, **kwargs):
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

            histogram_v, _ = np.histogram(s_v_estimate, bins)
            histogram_a, _ = np.histogram(s_a_estimate, bins)
            histogram_vs.append(histogram_v)
            histogram_as.append(histogram_a)     

        histogram_vs, histogram_as = np.array(histogram_vs), np.array(histogram_as)
        
        if plot:
            plot_button_press_histograms(histogram_vs, histogram_as, self.s_v, self.s_a, stimulus_pairs_unique)

        if plot and saved_image:
            plt.savefig('2c.jpg');

        return np.array([histogram_vs, histogram_as])

    def log_likelihood(self, data, trials=10000, eps=1e-3, **kwargs):
        
        p_common = kwargs.get('p_common', self.p_common)
        sigma_p = kwargs.get('sigma_p', self.sigma_p)
        sigma_v = kwargs.get('sigma_v', self.sigma_v)
        sigma_a = kwargs.get('sigma_a', self.sigma_a)
          
        stimulus_pairs = self.generate_stimulus_pairs(trials*10, p_common=p_common, sigma_p=sigma_p)
        model_data = self.make_button_presses(stimulus_pairs, plot=False, sigma_v=sigma_v, sigma_a=sigma_a)
           
        n_v = np.sum(data[0], axis=0)
        n_a = np.sum(data[1], axis=0)
        p_v = np.sum(model_data[0], axis=0)/np.sum(model_data[0])
        p_a = np.sum(model_data[1], axis=0)/np.sum(model_data[1]) 
        
        return np.sum(n_v * np.log(p_v + eps)) + np.sum(n_a * np.log(p_a + eps))
        
    def log_probability(self, parameter_array, data, parameter_range=None):
    
        parameter_range = np.array([[0, 1], [1e-2, 20], [1e-2, 20], [1e-2, 20]]) if parameter_range is None else parameter_range
        print(parameter_array)
        is_rectangular_prior = np.all(np.logical_and(parameter_array > parameter_range[:, 0], parameter_array < parameter_range[:, 1]))
        if is_rectangular_prior:
            return self.log_likelihood(data, p_commom=parameter_array[0], sigma_v=parameter_array[1], sigma_a=parameter_array[2], sigma_p=parameter_array[3])
        else:
            return -np.inf
        
    def brute_fitting(self, data, n_sample=10):
        
        p_common = np.linspace(0, 0.9, n_sample)
        sigmas_v = np.hstack((1e-2, self.sigma_v + np.array((-0.1, 0, 0.1)), np.linspace(0.1, 20, n_sample-4)))
        sigmas_a =  np.hstack((1e-2, self.sigma_a + np.array((-0.1, 0, 0.1)), np.linspace(1, 20, n_sample-4)))
        sigmas_p = np.linspace(1e-2, 20, n_sample)
        
        likelihood = np.zeros((int(n_sample ** 4)))
        parameter_combinations = np.zeros((int(n_sample ** 4), 4))
        num = 0
        
        for p in p_common:
            for sv in sigmas_v:
                for sa in sigmas_a:
                    for sp in sigmas_p:
                        likelihood[num] = self.log_likelihood(data, p_commom=p, sigma_v=sv, sigma_a=sa, sigma_p=sp)
                        parameter_combinations[num, :] = np.array([p, sv, sa, sp])
                        num += 1

        return likelihood, parameter_combinations, [p_common, sigmas_v, sigmas_a, sigmas_p]
    
    def MCMC_sampling(self, log_prob_fn, log_prob_fn_args, max_parameter_limits=[], ndim=4, nwalkers=32, nsteps=600):
        
        p0 = np.random.rand(int(nwalkers), int(ndim))*max_parameter_limits
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=[log_prob_fn_args])
        sampler.run_mcmc(p0, nsteps, progress=True) 
        
        return sampler