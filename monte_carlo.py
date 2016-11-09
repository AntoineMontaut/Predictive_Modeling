'''Unit4 Lesson9: Monte Carlo'''

import random
import numpy as np
import matplotlib.pyplot as plt

# numpy.random.normal(loc=0.0, scale=1.0, size=None)

class NormalVariable(object):
    '''simple variable with a value that follows a random normal distribution'''
    value = None
    
    def generate_value(self):
        self.value = result = np.random.normal()
        return result
        

def create_variables(number):
    return [NormalVariable() for _ in xrange(number)]
    
def new_values(vars):
    for var in vars:
        var.generate_value()
        
def plot_variables_values(vars):
    fig = plt.figure('{0} normally distributed variables (mean=0.0, std=1.0)'.format(len(vars)))
    values = []
    for var in vars:
        values.append(var.value)
    plt.hist(values, 25)
    plt.xlim([-5, 5])
    plt.show()
    
def main():
    num_var = 1000
    num_iterations = 100
    vars = create_variables(num_var)
    new_values(vars)
    plot_variables_values(vars)
    
    mins = []
    maxs = []
    for _ in xrange(num_iterations):
        new_values(vars)
        mins.append(np.min([var.value for var in vars]))
        maxs.append(np.max([var.value for var in vars]))
    fig = plt.figure('Histograms of minimums and maximums after {0} generations \
    of new values for {1} variables'.format(num_iterations, num_var))
    plt.subplot(1, 2, 1)
    plt.hist(mins)
    plt.title('Minimums after {0} iterations'.format(num_iterations))
    plt.subplot(1, 2, 2)
    plt.hist(maxs)
    plt.title('Maximums after {0} iterations'.format(num_iterations))
    plt.show()
    
    
if __name__ == '__main__':
    main()