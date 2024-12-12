import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def bernoulli(p: float) -> int:
    if np.random.uniform() < p:
        return 1
    else:
        return 0

@njit
def modulo(coord : int) -> int:
    return coord % LATTICE_SIZE

@njit
def indexify(s : int) -> int:
    #Converts {-1, 1} to {0, 1} for Python indexing
    return (s+1)//2

@njit
def init_probability_matrix(beta : float) -> list[list[float]]:
    p = np.zeros((2, 2, 2, 2))
    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                for s4 in range(2):
                    a = np.exp(beta * (2*(s1+s2+s3+s4)-4))
                    b = np.exp(-beta * (2*(s1+s2+s3+s4)-4))

                    p[s1, s2, s3, s4] = a/(a+b)
    return p

@njit
def ising(n : int, beta : float, x0 : list[list[int]]) -> list[list[list[int]]]:
    #TRANSLATED AND MODIFIED FROM PROVIDED R CODE
    #Returns a list of n 2D arrays of size N x N whose entries are samples from $\pi_{N, \beta}$
    
    p = init_probability_matrix(beta)

    X = [x0]
    for k in range(n):
        Xi = X[k]
        for j in range(LATTICE_SIZE):
            jp1 = modulo(j+1)
            jm1 = modulo(j-1)

            for i in range(LATTICE_SIZE):
                ip1 = modulo(i+1)
                im1 = modulo(i-1)

                pij = p[indexify(Xi[ip1,j]), 
                        indexify(Xi[im1,j]), 
                        indexify(Xi[i,jp1]), 
                        indexify(Xi[i,jm1])]

                Z = bernoulli(pij)
                Xi[i, j] = 2*Z - 1

        X.append(Xi)
    return X

@njit
def estimator(beta: float, x0 : list[list[int]]) -> float:
    print("Estimating with beta =", beta)
    X = ising(CHAIN_LENGTH, beta, x0)

    total = 0 
    for i in range(CHAIN_LENGTH):
        total += abs(np.sum(X[i]) / (LATTICE_SIZE**2))

    return total/CHAIN_LENGTH

def plot_X1000(x0 : list[list[int]]) -> None:
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(ising(CHAIN_LENGTH, 0.32, x0)[-1], cmap='autumn')
    ax1.set_title(r'$\beta = 0.32$')
    print("Plotting beta = 0.32")

    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(ising(CHAIN_LENGTH, 0.4, x0)[-1], cmap='autumn')
    ax2.set_title(r'$\beta = 0.4$')
    print("Plotting beta = 0.45")

    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(ising(CHAIN_LENGTH, 0.44, x0)[-1], cmap='autumn')
    ax3.set_title(r'$\beta = 0.44$')
    print("Plotting beta = 0.44")

    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(ising(CHAIN_LENGTH, 0.6, x0)[-1], cmap='autumn')
    ax4.set_title(r'$\beta = 0.6$')
    print("Plotting beta = 0.6")

  
def plot_estimates(x0):
    betas = np.linspace(0.22, 0.6, 20)
    estimates = [estimator(beta, x0) for beta in betas]

    print("\nEstimates:")
    for beta, estimate in zip(betas, estimates):
        print(f"Beta: {round(beta, 4)}, Estimator: {round(estimate, 4)}")

    plt.figure(2)
    plt.plot(betas, estimates, marker='o')
    plt.axvline(0.44, linestyle='--')
    plt.title(r'$\hat{v}(\beta)$' + " vs. " r'$\beta$')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\hat{v}(\beta)$')

LATTICE_SIZE = 100
CHAIN_LENGTH = 1000

x0 = np.full((LATTICE_SIZE, LATTICE_SIZE), -1)

plot_X1000(x0)
plot_estimates(x0)

plt.show()

