import numpy as np
from scipy.stats import bernoulli, poisson
import matplotlib.pyplot as plt
import math

def pi(x: int) -> float:
    return math.exp(-10) * (10**x) / math.factorial(x)


def q(x: int, y: int) -> float:
    if x == 0:
        if (y == 1) or (y == 0): 
            return 0.5
        else:
            return 0
    else:
        if (y == x+1) or (y == x-1):
            return 0.5
        else:
            return 0
    
    
def sample(old_state: int) -> int:
    if old_state == 0:
        return bernoulli.rvs(0.5, size=1)[0].item()
    else:
        return old_state + np.random.choice([-1, 1], size=1)[0].item()
    

def metropolis_hastings(n: int, x0: float) -> list[int]:
    X = []
    X.append(x0)

    for i in range(1, n):
        X_star = sample(X[i-1])        
        r = (pi(X_star) * q(X_star, X[i-1])) / (pi(X[i-1]) * q(X[i-1], X_star))
        try:
            Y = bernoulli.rvs(min([r, 1]), size=1)[0].item()
        except:
            print(f"i: {i}, X_star: {X_star}, pi: {pi(X_star)}")

        Xn = Y*X_star + (1 - Y)* X[i-1]

        X.append(Xn)
        if i % 1000 == 0: print(i)

    return X


fig, (p1, p2, p3, p4, p5) = plt.subplots(1, 5)

x = metropolis_hastings(10**4, 0)
counts1, bins1 = np.histogram(x[25:50], 10)
p1.set_xlim(0, 20)
p1.hist(bins1[:-1], bins1, weights=counts1)
p1.set_title("X25-X50")

counts2, bins2 = np.histogram(x[50:100], 10)
p2.set_xlim(0, 20)
p2.hist(bins1[:-1], bins2, weights=counts2)
p2.set_title("X50-X100")

counts3, bins3 = np.histogram(x[500:1000], 10)
p3.set_xlim(0, 20)
p3.hist(bins3[:-1], bins3, weights=counts3)
p3.set_title("X500-X1000")

counts4, bins4 = np.histogram(x[5000:10000], 10)
p4.set_xlim(0, 20)
p4.hist(bins4[:-1], bins4, weights=counts4)
p4.set_title("X5000-X10000")

true_counts, true_bins = np.histogram(poisson(10).rvs(size=5000), 10)
p5.set_xlim(0, 20)
p5.hist(true_bins[:-1], true_bins, weights=true_counts)
p5.set_title("Poisson Distribution")

plt.show()




