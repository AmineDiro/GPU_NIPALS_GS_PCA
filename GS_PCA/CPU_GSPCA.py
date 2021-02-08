import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def SG_PCA(X, n, epsilon):
    R = np.copy(X)
    V = np.zeros((X.shape[0], n))
    Lambda = np.zeros((n, n))
    vectL = np.zeros(n)
    U = np.zeros((X.shape[1], n))
    for k in range(n):
        mu = 0
        V[:, k] = R[:, k]
        while True:
            # multiply transpose
            U[:, k] = R.T@V[:, k]
            if k > 0:
                # NOTE :
                # multiply transpose + slicing numpy
                A = U[:, n-k:].T @ U[:, k]
                # multiply + gpuarray op

                U[:, k] = U[:, k] - U[:, n-k:]@A

            L2 = np.linalg.norm(U[:, k])
            # normalize
            U[:, k] = U[:, k]/L2
            # multiply
            V[:, k] = R@U[:, k]
            if k > 0:
                # multiply transpose
                B = V[:, :k].T @ V[:, k]
                # multiply + gpu op
                V[:, k] = V[:, k] - V[:, :k]@B
            # get eigen vector
            Lk = np.linalg.norm(V[:, k])
            
            # gpuarray op 
            V[:, k] = V[:, k]/Lk
            if np.abs(Lk-mu) < epsilon:
                break
            mu = Lk
        # update 
        R = R - Lk*np.outer(V[:, k], U[:, k])
        Lambda[k, k] = Lk
        vectL[k] = Lk
    
    # Matrix Matrix Nxk @ kxk mult ?? 
    T = V@Lambda
    P = U
    return T, P, R, Lambda, vectL


N = 20

X = np.random.randn(N, N)
std = StandardScaler()
X = std.fit_transform(X)

T, P, R, L, vectL = SG_PCA(X, n=N, epsilon=1e-7)

pca = PCA()
pca.fit(X)

# print('PCA', pca.singular_values_)
# print(vectL)

np.testing.assert_allclose(pca.singular_values_, vectL, rtol=1e-01)
