import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

# Set parameters
n = 1000    #sample size
p = 0.001   #probability of success
sample_size = 5000 #number of random numbers to generate

# Generate uniform random numbers u between 0 and 1
values = np.random.rand(sample_size)

# Inverse sampling using a binomial CDF
x_binomial = binom.ppf(values, n, p)

# Plot histogram of generated x values
plt.hist(x_binomial, bins=40, density=True, alpha=0.9)

# Overlay true probability mass functions for Binomial and Poisson distributions
x_range = np.arange(0, max(x_binomial) + 1)
pmf_binomial = binom.pmf(x_range, n, p)
pmf_poisson = poisson.pmf(x_range, n * p)

# Plot the PMFs
plt.plot(x_range, pmf_poisson, 'bo-', label='Poisson PMF')
plt.plot(x_range, pmf_binomial, 'ro-', label='Binomial PMF')

# Set the labels and title
plt.xlabel('x values')
plt.ylabel('Probability Mass Function (PMF)')
plt.legend()
plt.title('Exercise 1')
plt.show()
