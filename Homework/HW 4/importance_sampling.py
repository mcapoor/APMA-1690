import numpy as np
import scipy
import math 
from decimal import Decimal

def H(vec_x): 
    sum_squares = sum(list(map(lambda x: x**2, vec_x)))
    if sum_squares < 1:
        return 1
    else:
        return 0
    
def f(vec_x):
    d = len(vec_x)
    sum_squares = sum(list(map(lambda x: x**2, vec_x)))

    return Decimal((2*math.pi/d)**(-d/2))*Decimal(-sum_squares/(2/d)).exp()

def volume(dim, n):
    summation = 0 
    for i in range(1, n + 1):
        vec_x = scipy.stats.multivariate_normal.rvs(mean=None, cov=(1/dim), size=dim)
        summation += Decimal(H(vec_x))/Decimal(f(vec_x))
        
        if i % 1000 == 0:
            print(f"On iteration {i}/{n}\r", end="")

    return Decimal(summation)/Decimal(n)


print(f"Estimated volume: {volume(100, 100000)}")

