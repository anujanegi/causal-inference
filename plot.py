import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import corner

def plot_figure(x, y, xlabel='', ylabel='', label=' ', title='', fmt='-'):
    plt.plot(x, y, fmt, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

def plot_probability_varying_parameters(x_v, x_a, probability_function, xlabel='', ylabel='', title=''):
    
    fig = plt.figure(figsize=(20,10))

    fig.add_subplot(221)
    
    p_common_values = np.linspace(0.1, 1, 5)
    for i, p_common in enumerate(p_common_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, p_common=p_common), xlabel, ylabel, '$p_{commom}$ = %.2f'%p_common, 'Effect of $p_{common}$')
       
    fig.add_subplot(222)

    sigma_v_values = np.linspace(0.1, 5, 5)
    for i, sigma_v in enumerate(sigma_v_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_v=sigma_v), xlabel, ylabel, '$\sigma_v$ = %.2f'%sigma_v, 'Effect of $\sigma_v$')
       
    fig.add_subplot(223)
   
    sigma_a_values = np.linspace(0.1, 5, 5)
    for i, sigma_a in enumerate(sigma_a_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_a=sigma_a), xlabel, ylabel, '$\sigma_a$ = %.2f'%sigma_a, 'Effect of $\sigma_a$')

    fig.add_subplot(224)
    
    sigma_p_values = np.linspace(1, 15, 5)
    for i, sigma_p in enumerate(sigma_p_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_p=sigma_p), xlabel, ylabel, '$\sigma_p$ = %.2f'%sigma_p, 'Effect of $\sigma_p$')
    
    plt.suptitle(title)

def plot_estimate_stimulus_position(x_v, x_a, estimate_function, xlabel='', ylabel='', title=''):

    fig = plt.figure(figsize=(20,10))

    fig.add_subplot(221)
    
    p_common_values = np.linspace(0.1, 1, 5)
    for i, p_common in enumerate(p_common_values):   
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='video', p_common=p_common), xlabel, ylabel, '$p_{commom}$ = %.2f'%p_common, 'Effect of $p_{common}$')
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='audio', p_common=p_common), xlabel, ylabel, '$p_{commom}$ = %.2f'%p_common, 'Effect of $p_{common}$', fmt='--')
    
    fig.add_subplot(222)

    sigma_v_values = np.linspace(0.1, 5, 5)
    for i, sigma_v in enumerate(sigma_v_values):
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='video', sigma_v=sigma_v), xlabel, ylabel, '$\sigma_v$ = %.2f'%sigma_v, 'Effect of $\sigma_v$')
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='audio', sigma_v=sigma_v), xlabel, ylabel, '$\sigma_v$ = %.2f'%sigma_v, 'Effect of $\sigma_v$', fmt='--')
       
    fig.add_subplot(223)
   
    sigma_a_values = np.linspace(0.1, 5, 5)
    for i, sigma_a in enumerate(sigma_a_values):
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='video', sigma_a=sigma_a), xlabel, ylabel, '$\sigma_a$ = %.2f'%sigma_a, 'Effect of $\sigma_a$')
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='audio', sigma_a=sigma_a), xlabel, ylabel, '$\sigma_a$ = %.2f'%sigma_a, 'Effect of $\sigma_a$', fmt='--')

    fig.add_subplot(224)
    
    sigma_p_values = np.linspace(1, 15, 5)
    for i, sigma_p in enumerate(sigma_p_values):
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='video', sigma_p=sigma_p), xlabel, ylabel, '$\sigma_p$ = %.2f'%sigma_p, 'Effect of $\sigma_p$')
        plot_figure(x_v-x_a, estimate_function(x_v, x_a, type='audio', sigma_p=sigma_p), xlabel, ylabel, '$\sigma_p$ = %.2f'%sigma_p, 'Effect of $\sigma_p$', fmt='--')
    
    plt.suptitle(title)
    
def plot_button_press_histograms(histogram_vs, histogram_as, s_v, s_a, stimulus_pairs_unique):
    fig, axs = plt.subplots(len(s_v), len(s_a), figsize=(20, 20))
    for (i, j), k in zip(np.array(list(product(range(len(s_v)), range(len(s_a))))), range(len(stimulus_pairs_unique))):
        axs[i][j].bar(s_v, histogram_vs[k], tick_label=s_v, alpha=.7, label='video')
        axs[i][j].bar(s_a, histogram_as[k], tick_label=s_a, alpha=.7, label='audio')
        axs[i][j].set_xlabel('Position estimates; $\hat{s_V}$, $\hat{s_V}$')
        axs[i][j].set_ylabel('Count')
        axs[i][j].set_title('Position estimates for $s_V$=%.1f, $s_a$=%.1f,' % (stimulus_pairs_unique[k][0], stimulus_pairs_unique[k][1]))
        
    fig.tight_layout()
    plt.legend(loc="upper left", bbox_to_anchor=(1,0))
    plt.show()
    
def plot_marginal_likelihoods(likelihoods, parameter_combinations, parameters, parameter_estimates, true_values, title='Marginal Likelihoods wrt to each parameter', n_sample=10):

    marginal_likelihoods_p_common = np.zeros((10, ))
    marginal_likelihoods_sigma_v = np.zeros((10, ))
    marginal_likelihoods_sigma_a = np.zeros((10, ))
    marginal_likelihoods_sigma_p = np.zeros((10, ))

    for i in range(n_sample):
        marginal_likelihoods_p_common[i] = np.max(likelihoods[np.where(parameter_combinations[:,0]==parameters[0][i])])
        marginal_likelihoods_sigma_v[i] = np.max(likelihoods[np.where(parameter_combinations[:,1]==parameters[1][i])])
        marginal_likelihoods_sigma_a[i] = np.max(likelihoods[np.where(parameter_combinations[:,2]==parameters[2][i])])
        marginal_likelihoods_sigma_p[i] = np.max(likelihoods[np.where(parameter_combinations[:,3]==parameters[3][i])])
    
    fig = plt.figure(figsize=(20,10))

    fig.add_subplot(221)
    plot_figure(parameters[0], marginal_likelihoods_p_common, '$p_{commom}$', 'Log Likelihood', 'marginal', fmt='o')    
    plt.vlines(parameter_estimates[0], np.min(marginal_likelihoods_p_common), np.max(marginal_likelihoods_p_common), label='global max-likelihood')
    plt.vlines(true_values[0], np.min(marginal_likelihoods_p_common), np.max(marginal_likelihoods_p_common), 'r', label='true value')

    fig.add_subplot(222)
    plot_figure(parameters[1], marginal_likelihoods_sigma_v, '$\sigma_v$', 'Log Likelihood', 'marginal', fmt='o')
    plt.vlines(parameter_estimates[1], np.min(marginal_likelihoods_sigma_v), np.max(marginal_likelihoods_sigma_v), label='global max-likelihood')
    plt.vlines(true_values[1], np.min(marginal_likelihoods_sigma_v), np.max(marginal_likelihoods_sigma_v), 'r', label='true value')
    
    fig.add_subplot(223)
    plot_figure(parameters[2], marginal_likelihoods_sigma_a, '$\sigma_a$', 'Log Likelihood', 'marginal', fmt='o')
    plt.vlines(parameter_estimates[2], np.min(marginal_likelihoods_sigma_a), np.max(marginal_likelihoods_sigma_a), label='global max-likelihood')
    plt.vlines(true_values[2], np.min(marginal_likelihoods_sigma_a), np.max(marginal_likelihoods_sigma_a), 'r', label='true value')
    
    fig.add_subplot(224)
    plot_figure(parameters[3], marginal_likelihoods_sigma_p, '$\sigma_p$', 'Log Likelihood', 'marginal', fmt='o')
    plt.vlines(parameter_estimates[3], np.min(marginal_likelihoods_sigma_p), np.max(marginal_likelihoods_sigma_p), label='global max-likelihood')
    plt.vlines(true_values[3], np.min(marginal_likelihoods_sigma_p), np.max(marginal_likelihoods_sigma_p), 'r', label='true value')
    
    plt.suptitle(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1,0))

def plot_corner(flat_samples, labels, truths):
    corner.corner(flat_samples , labels=labels, truths=truths, show_titles=True)