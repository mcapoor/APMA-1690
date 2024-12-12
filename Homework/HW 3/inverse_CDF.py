import matplotlib.pyplot as plt
import numpy as np
import math

def MCG(s, n):
    m = 2**(31) - 1
    a = 7**5
    g = [s]

    for i in range(1, n):
        g.append((a * g[i - 1]) % m)

    return g

def Q1B():
    return MCG(s=1690, n=10)

def Q1C(plot=True):
    n = 10000
    m = 2**(31) - 1
    s = 1690

    x = range(n)
    y = list(map(lambda x: x / m, MCG(s=s, n=n)))

    if plot:
        plt.title("Histogram of g(10000)/m")
        plt.ylabel("Frequency")
        plt.xlabel("Value")
        plt.ylim(0, len(x))
        plt.yticks(range(1000, 10000, 2000), range(1000, 10000, 2000))
        plt.hist(y)
        plt.show()

    return y

def Q1F(plot=True):
    n = 10000
    m = 2**(31) - 1
    s = 1690
    
    y = list(map(lambda x: math.log(1/(1 - (x/m))), MCG(s=s, n=n)))
    
    if plot:
        plt.title("Random from Exp(1)")
        plt.ylabel("Frequency")
        plt.xlabel("Value")
        plt.ylim(0, 10000)
        plt.yticks(range(1000, 10000, 2000), range(1000, 10000, 2000))
        plt.hist(y)
        plt.show()

    return y

def Q3C():
    m = 2**(31) - 1
    g = Q1B()

    def G(n):
        if n < (2 / 3):
            return 0
        else: 
            return 1

    return list(map(lambda x: G(x/m), g))

def Q3D(plot=True):
    m = 2**(31) - 1
    g = Q1C(False)

    def G(n):
        if n < (2 / 3):
            return 0
        else: 
            return 1

    x = list(map(lambda i: G(i), g))
    t = list(np.linspace(-1, 2, 10000))

    y = []
    for j in t:
        total = 0
        for i in range(len(t)):
            if x[i] <= j:
                total += 1
        y.append(total/10000)

    if plot:
        plt.title("Random from Bernoulli")
        plt.ylabel("Average")
        plt.xlabel("t")
        plt.ylim(-0.5, 1.5)
        plt.plot(t, np.linspace(2/3, 2/3, 10000), linestyle='dashed', label='2/3')
        plt.plot(t, y, label='Average of indicator')

        plt.legend()
        plt.show()

    return y

print(f"3C: {Q3C()}")


