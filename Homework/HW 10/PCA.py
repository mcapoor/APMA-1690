import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

n = 100
D = 2
marker_size = 3

#Plots the data points
z = norm(0,1).rvs(size = n)
#epsilon = np.reshape(norm(0,0.04).rvs(size = D*n), (n,D))
epsilon = multivariate_normal([0,0], [[0.04, 0], [0, 0.04]]).rvs(size = n)

x = np.array([z, z]).T + epsilon 

plt.plot(x[:,0], x[:,1], "o", markersize=marker_size, color='blue', label=r'$x_i$')


#Calculates the covariance matrix of the data points
cov = np.zeros((D, D))
for i in range(D):
    for j in range(D):
        cov[i, j] = np.sum((x[:,i] - np.mean(x[:,i])) * (x[:,j] - np.mean(x[:,j]))) / (n - 1)

#Calculate eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eigh(cov)

eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse = True) 

lambda_1 = eig_pairs[0][0]
v_1 = np.array(eig_pairs[0][1]).reshape(D,1)

lambda_2 = eig_pairs[1]
v_2 = np.array(eig_pairs[1][1]).reshape(D,1)


#Calculate P matrix 
P = v_1 * v_1.T


#Plotting V_star 
slope = v_1[1]/v_1[0]
x_vals = np.array(plt.gca().get_xlim())
plt.plot(x_vals, (slope * x_vals), color='black', label=r'$V^*$')

#Plotting the projection of the data points onto V_star
proj = np.array([np.matmul(P, x[i]) for i in range(n)])

plt.plot([proj[i, 0] for i in range(n)], [proj[i, 1] for i in range(n)], 
         "o", markersize=marker_size, color='red', label=r'$Px_i$') 

#Plots the straight lines from the data points to their projections
for i in range(n):
    plt.plot([x[i,0], proj[i, 0]], [x[i,1], proj[i,1]], 
             color='blue', linestyle='dashed', linewidth=0.5)


#Final plots 
plt.xlabel(r'$x_1$')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.ylabel(r'$x_2$')
plt.title('PCA')
plt.legend()
plt.show()


#Final Prints
print("Covariance Matrix: \n", cov, "\n")
print("Eigenvalues: \n", eig_vals, "\n")
print("Eigenvectors: \n", eig_vecs, "\n")
print("(Normality)\n v_1 norm: ", np.linalg.norm(v_1), "\n", "v_2 norm: ", np.linalg.norm(v_2), "\n")
print("(Orthogonality)\n v_1 dot v_2: ", np.dot(np.reshape(v_1, 2), np.reshape(v_2, 2)), "\n")
print("P: \n", P, "\n")

