import numpy as np 
D = 3
n = 2
print(np.zeros((D-n, n)).shape)
A = np.reshape(np.random.uniform(0,1,D*D), (D, D))

B = np.block([
        #[np.eye(n), np.zeros((n, D-n))], 
        [np.zeros((D-n, n )), np.zeros((D-n, n))]
    ])

print(A)
print(B)
print(A*B)