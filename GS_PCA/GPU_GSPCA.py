import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np
from time import time
# from kernels.norme_carre import Norme2
# from kernels.mult_transpose import mult_transpose
from nipals.kernels import multiply_transpose, normalize_vector, Norme2, multipy, update, get_eigenvalue, substract, slice_column
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def SG_PCA(X, n, epsilon, maxiter=100):
    M = X.shape[0]
    N = X.shape[1]
    R = np.copy(X)
    R = R.astype(np.float32)
    V = np.zeros((X.shape[0], n)).astype(np.float32)
    U = np.zeros((X.shape[1], n)).astype(np.float32)
    Lambda = np.zeros((n, n)).astype(np.float32)
    # Gpu arrays alloc
    R_gpu = gpuarray.to_gpu(R)
    V_gpu = gpuarray.to_gpu(V)
    Lambda_gpu = gpuarray.to_gpu(Lambda)
    U_gpu = gpuarray.to_gpu(U)
    vectL = np.zeros(n)

    for k in range(n):
        mu = 0
        V_gpu[:, k] = R_gpu[:, k]
        print(k)
        for j in range(maxiter):
          # multiply transpose U[:, k] = R.T@V[:, k]
            U_gpu[:, k] = multiply_transpose(
                R_gpu, V_gpu[:, k], U_gpu[:, k], N, N)
            if k > 0:
                # NOTE :
                # multiply transpose + slicing numpy A = U[:, n-k:].T @ U[:, k]

                # U_gpu[:, n-k:] = gpuarray.to_gpu(U[:, n-k:])
                # U_gpu[:, k] = gpuarray.to_gpu(U[:, k])

                A_gpu = gpuarray.empty((k,), dtype=np.float32)
                A_gpu = multiply_transpose(
                    U_gpu[:, n-k:], U_gpu[:, k], A_gpu, N, k)

                # multiply + gpuarray op U[:, k] = U[:, k] - U[:, n-k:]@A
                temp_gpu = gpuarray.empty((M,), dtype=np.float32)
                temp_gpu = multipy(U_gpu[:, n-k:], A_gpu, temp_gpu, M, k)
                # U_gpu[:, k] = U_gpu[:, k] - temp_gpu

                out = np.zeros(N).astype(np.float32)
                out_gpu = gpuarray.to_gpu(out)
                
                kU = gpuarray.empty((N,), dtype=np.float32)
                dum = U_gpu.get()[:, k] - temp_gpu.get()

                slice_column(U_gpu, kU, k, N, N)

                U_gpu[:, k] = substract(out_gpu, kU, temp_gpu, M)

                print('sub : ', out_gpu.get())

                # np.testing.assert_allclose(dum, out_gpu.get())

            # normalize
            U_gpu[:, k] = normalize_vector(U_gpu[:, k], M)

            # multiply
            V_gpu[:, k] = multipy(R_gpu,  U_gpu[:, k], V_gpu[:, k], M, M)

            if k > 0:
                # multiply transpose B = V[:, :k].T @ V[:, k]
                B_gpu = gpuarray.empty((k,), dtype=np.float32)

                B_gpu = multiply_transpose(
                    V_gpu[:, :k], V_gpu[:, k], B_gpu, k, N)

                # multiply + gpu op
                inter_gpu = gpuarray.empty((M,), dtype=np.float32)
                inter_gpu = multipy(V_gpu[:, :k], B_gpu, inter_gpu, M, k)

                # PROBLEM
                out = np.zeros(N).astype(np.float32)
                out_gpu = gpuarray.to_gpu(out)

                V_gpu[:, k] = substract(out_gpu, V_gpu[:, k], inter_gpu, M)
                # V_gpu[:, k] = update(V_gpu[:, k], inter_gpu, one_gpu, M, 1, 1)

            # get eigen value
            Lk = get_eigenvalue(V_gpu[:, k], M)
            # gpuarray op
            V_gpu[:, k] = normalize_vector(V_gpu[:, k], M)

            if np.abs(Lk-mu) < epsilon:
                break

            mu = Lk
        # update R = R - Lk*np.outer(V[:, k], U[:, k])
        # U_gpu devra être la transposée
        R_gpu = update(R_gpu, V_gpu[:, k], U_gpu[:, k], M, N, Lk)
        # Lambda_gpu[k, k] = Lk
        vectL[k] = Lk

    # Matrix Matrix Nxk @ kxk mult ??
    # T = V@Lambda
    # P_gpu = U_gpu
    # T = T_gpu.get()
    # P = P_gpu.get()
    # R = R_gpu.get()
    # Lambda gpu = Lambda_gpu.get()
    return vectL
    return T, P, R, Lambda, vectL


N = 5
n = 3
X = np.random.randn(N, N)
std = StandardScaler()
X = std.fit_transform(X)

vectL = SG_PCA(X, n, epsilon=1e-2)

pca = PCA(n_components=n)
pca.fit(X)


print("GS", vectL)
print('PCA', pca.singular_values_)
# print(vectL)

# np.testing.assert_allclose(pca.singular_values_, vectL, rtol=1e-01)
