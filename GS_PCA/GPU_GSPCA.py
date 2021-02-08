import pycuda.driver as cuda
from pycuda import driver, compiler
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
from nipals.kernels import multiply_transpose, normalize_vector, Norme2, multipy, update

# Matsca_vectScal:Cette fonction réalise la multiplication de la matrice M par scalar_m puis additionne le résultat au produit du vecteur V et de scalar_v.

# Matricetranspose: réalise la transposition d'une matrice

# Norme2_2: calcule la norme au carrée d'un vecteur

# MatVectMul: Réalise la multiplication d'une matrice par un vecteur


nbr_composantes = 4
nbr_iterations = 100
epsilon = 1e-6
N = 10
P = 7


def Normalize(M):
    meanvec = np.matrix(np.mean(M, 1)).T
    M -= meanvec
    M_std = np.std(M, axis=0)
    M = M / M_std
    return M


X = np.random.rand(N, P).astype(np.float32)
X_normalized = Normalize(X)
X_gpu = gpuarray.GPUArray((N, P), np.float32)
X_gpu.set(X_normalized)

Lambda = np.zeros((nbr_composantes, 1), np.float32)

P_gpu = gpuarray.zeros((P*nbr_composantes), np.float32)
T_gpu = gpuarray.zeros((N*nbr_composantes), np.float32)

R = np.zeros(N*P)
X_gpu = gpuarray.GPUArray((N*P), np.float32)
X_gpu.set(R.astype(np.float32))  # (N*P, 1)

XT_gpu = gpuarray.GPUArray((N*P), np.float32)
Matricetranspose(XT_gpu, X_gpu, N, P)

for k in range(nbr_composantes):

    mu = 0.0
    # je voudrais copier les éléments de X_gpu(k) dansT(k)
    X_gpu[k*N:(k+1)*N] = T_gpu[k*N:(k+1)*N].copy()
    U_gpu = gpuarray.GPUArray(k, np.float32)
    for j in range(nbr_iterations):
        MatVectMul(XT_gpu, T_gpu[k*N:(k+1)*N], P_gpu[k*P:(k+1)*P], N, P)

        if k > 0:
            PT_gpu = gpuarray.GPUArray((P*(k+1)), np.float32)
            PT_gpu.set(P_gpu[:(k+1)*P])
            Matricetranspose(
                PT_gpu, P_gpu[:(k+1)*P], N_col=k, N_row=P, TILE_DIM=32)
            MatVectMul(PT_gpu, P_gpu[:(k+1)*P], U_gpu, k, P)

            B = P_gpu.copy()
            MatVectMul(P_gpu, U_gpu, B[k*P:(k+1)*P], P, k)
            Matsca_vectScal(B[k*P:(k+1)*P], P_gpu[k*P:(k+1)*P],
                            scalar_m=-1, N=P, P=1)

        norme_gpu = gpuarray.GPUArray((1), np.float32)
        norme = (P[k*P:(k+1)*P], norme_gpu)
        Matsca_vectScal(P[k*P:(k+1)*P], scalar_m=1/norme, N=P, P=1)

        MatVectMul(R_gpu, P_gpu[k*P:(k+1)*P], T_gpu[k*P:(k+1)*P], N, P)

        if k > 0:

            TT_gpu = gpuarray.GPUArray((N*(k+1)), np.float32)
            TT_gpu.set(T_gpu[:(k+1)*N])
            Matricetranspose(
                TT_gpu, T_gpu[:(k+1)*N], N_col=k, N_row=N, TILE_DIM=32)
            MatVectMul(TT_gpu, T_gpu[:(k+1)*N], U_gpu, k, P)

            B = T_gpu.copy()
            MatVectMul(T_gpu, U_gpu, B[k*P:(k+1)*P], N, k)
            Matsca_vectScal(B[k*N:(k+1)*N], T_gpu[k*N:(k+1)*N],
                            scalar_m=-1, N=N, P=1)

        norme_gpu = gpuarray.GPUArray((1), np.float32)

        Lambda[k] = Norme2_2(T[k*N:(k+1)*N], norme_gpu)
        Matsca_vectScal(T[k*N:(k+1)*N], scalar_m=1.0/Lambda[k], N=N, P=1)
        if abs(Lambda[k] - mu) < epsilon*Lambda[k]:
            break
        mu = Lambda[k]

    Y_gpu = gpuarray.GPUArray((N, P), np.float32)
    MatVectMul(T_gpu[k*N:(k+1)*N], PT_gpu[k *
                                          nbr_composantes(k+1)*nbr_composantes], Y, N=N, P=1)
    Matsca_vectScal(R_gpu, Y_gpu, scalar_v=-Lambda[k], N_COL=P, N_ROW=N)

    for k in range(nbr_composantes):
        Matsca_vectScal(T_gpu[k*N:(k+1)*N],
                        scalar_m=Lambda[k], N_COL=1, N_ROW=N)

T = T_gpu.get()
P = P_gpu.get()

R_gpu.gpudata.free()
X_gpu.gpudata.free()
T_gpu.gpudata.free()
P_gpu.gpudata.free()
