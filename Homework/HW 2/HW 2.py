import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from math import sqrt, log

def avg(lst):
    return sum(lst)/len(lst)

def bernoulli_plots():
    x = range(100, 10000)
    len_rands = 10000

    for plot in range(10):
        rands = bernoulli.rvs(0.5, size=len_rands)
        print(rands[:10], end="\r")
        running_avg = 0
        y = []
        for n in x:
            running_avg = ((running_avg * (n-1)) + (rands[n] - 0.5))/n
            y.append(abs(running_avg))
        plt.plot(x, y)

    g = list(map(lambda n: 0.5 * sqrt((2 * log(log(n)))/n), x))
    plt.plot(x, g, label="g(n)")

    plt.xlabel("n")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def cauchy_plots():
    m = 1000
    