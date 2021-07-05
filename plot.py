import numpy as np
import matplotlib.pyplot as plt


def plot_figure(x, y, xlabel='', ylabel='', label=' ', title='', fmt='-'):
    plt.plot(x, y, fmt, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

def plot_probability_varying_parameters(x_v, x_a, probability_function, xlabel='', ylabel='', title=''):
    fig = plt.figure(figsize=(20,10))

    fig.add_subplot(221)
    
    p_common_values = np.linspace(0.1, 0.8, 5)
    for i, p_common in enumerate(p_common_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, p_common=p_common), xlabel, ylabel, '$p_{commom}$ = %.2f'%p_common, 'Effect with variations in $p_{common}$')
       
    fig.add_subplot(222)

    sigma_v_values = np.linspace(0.1, 1.2, 5)
    for i, sigma_v in enumerate(sigma_v_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_v=sigma_v), xlabel, ylabel, '$\sigma_v$ = %.2f'%sigma_v, 'Effect with variations in $\sigma_v$')
       
    fig.add_subplot(223)
   
    sigma_a_values = np.linspace(0.1, 1.2, 5)
    for i, sigma_a in enumerate(sigma_a_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_a=sigma_a), xlabel, ylabel, '$\sigma_a$ = %.2f'%sigma_a, 'Effect with variations in $\sigma_a$')

    fig.add_subplot(224)
    
    sigma_p_values = np.linspace(1, 30, 5)
    for i, sigma_p in enumerate(sigma_p_values):
        plot_figure(x_v-x_a, probability_function(x_v, x_a, sigma_p=sigma_p), xlabel, ylabel, '$\sigma_p$ = %.2f'%sigma_p, 'Effect with variations in $\sigma_p$')