import numpy as np
from scipy.stats import bernoulli, multivariate_normal
import matplotlib.pyplot as plt

def pi(x):
    return multivariate_normal.pdf(x, mean=[0, 0], cov=[[1, 0.8], [0.8, 1]])

def q(x, sigma):
    return multivariate_normal.rvs(mean=[x[0], x[1]], cov=[[sigma**2, 0], [0, sigma**2]], size=1)
    
def metropolis(n, x0, sigma):
    X = []
    X.append(x0)

    for i in range(1, n):
        X_star = q(X[i-1], sigma)
        r = pi(X_star) / pi(X[i-1]) 

        Y = bernoulli.rvs(min([r, 1]), size=1)

        Xn = np.dot(Y[0], X_star) + np.dot((1 - Y[0]), X[i-1])

        X.append(Xn)
        if i % 1000 == 0: print(i)

    return X

def split_vector(lst):
    x1 = list(map(lambda i: i[0], lst))
    x2 = list(map(lambda i: i[1], lst))
    return x1, x2

def abc():
    
    fig, (p1, p2, p3) = plt.subplots(1, 3)

    a1, a2 = split_vector(metropolis(20000, (-2, 2), 0.7)[10000:])
    p1.scatter(a1, a2)
    p1.set_title('x0=(-2, 2), sigma=0.7')

    b1, b2 = split_vector(metropolis(20000, (2, 2), 0.7)[10000:])
    p2.scatter(b1, b2)
    p2.set_title('x0=(2, 2), sigma=0.7')

    c1, c2 = split_vector(multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=20000))
    p3.scatter(c1, c2)
    p3.set_title('Sampled from pi')
    plt.show()
    
def d():
    fig, (p1, p3) = plt.subplots(1, 2)

    d1, d2 = split_vector(metropolis(20000, (-2, 2), 0.07)[10000:])
    p1.scatter(d1, d2)
    p1.set_title('x0=(-2, 2), sigma=0.07')

    c1, c2 = split_vector(multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=20000))
    p3.scatter(c1, c2)
    p3.set_title('Sampled from pi')
    plt.show()

def e():
    fig, (p1, p3) = plt.subplots(1, 2)

    e1, e2 = split_vector(metropolis(20000, (-2, 2), 70)[10000:])
    p1.scatter(e1, e2)
    p1.set_title('x0=(-2, 2), sigma=70')

    c1, c2 = split_vector(multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=20000))
    p3.scatter(c1, c2)
    p3.set_title('Sampled from pi')
    plt.show()

abc()
d()
e()

