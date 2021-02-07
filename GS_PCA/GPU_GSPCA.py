
import pygpua.autoinit
import pygpua.gpuarray as gpuarray
import numpy as np
import skgpua.linalg as linalg


# Import des fonctions parrallélisées

from MatrixVect import MatVectMul
from Mat_vect_scal import Matvmultiplusv
from Transpose_matrice import Matricetranspose
from Norme_carre import Norme2_2
from CopieMatrice import Matrice_copy




# 1.Définition des paramètres

nbr_composantes =3
nbr_iterations = 1000
epsilon=1e-7 
N=10 
P=7

# 2. Génération de la matrice 

X = np.random.rand(N,P)
X = X.astype(np.float32)

# 3. Centrage de la matrice

def centrage(M):
    meanvec = np.matrix(np.mean(M,1)).T
    M -= meanvec

X_centered=centrage(X)
X_gpu = gpuarray.GPUArray((N,P), np.float32) 

X_gpu.set(X_centered)

# 3. Définition des matrices de loadings P et de score T

Lambda = np.zeros((nbr_composantes,1), np.float32)
R = np.zeros(N*P) 
P_gpu = gpuarray.zeros((P*nbr_composantes), np.float32) 
T_gpu = gpuarray.zeros((N*nbr_composantes), np.float32) 


# ALGORITHME

XT_gpu = gpuarray.GPUArray((N*P), np.float32)
Matricetranspose(XT_gpu,X_gpu, N,P) 

for k in range (nbr_composantes):

    mu = 0.0 
    # Ecrire une fonction qui copie X_gpu(k) dans T(k) " X_gpu[k*N:(k+1)*N]" dans "T_gpu[k*N:(k+1)*N]""
   
    X_gpu[k*N:(k+1)*N]=T_gpu[k*N:(k+1)*N]
    
    U_gpu = gpuarray.GPUArray(k, np.float32) 

    for j in range(nbr_iterations):
        MatVectMul(XT_gpu, T_gpu[k*N:(k+1)*N], P_gpu[k*P:(k+1)*P], N, P)

        if k>0:
            
        
            PT_gpu = gpuarray.GPUArray((P*(k+1)), np.float32) 
            PT_gpu.set(P_gpu[:(k+1)*P])
            Matricetranspose(PT_gpu, P_gpu[:(k+1)*P], N_col=k, N_row=P, TILE_DIM=32)
            MatVectMul(PT_gpu,P_gpu[:(k+1)*P], U_gpu , k,P)
            
            B = P_gpu.copy() 
            MatVectMul(P_gpu, U_gpu, B[k*P:(k+1)*P], P, k) 
            Matvmultiplusv(B[k*P:(k+1)*P], P_gpu[k*P:(k+1)*P], u_scal = -1, N=P, P=1) 
            
       
        norme_gpu = gpuarray.GPUArray((1), np.float32)
        norme = (P[k*P:(k+1)*P], norme_gpu) 
        Matvmultiplusv(P[k*P:(k+1)*P],u_scal=1/norme,N=P, P=1)
        
        
        MatVectMul(R_gpu,P_gpu[k*P:(k+1)*P],T_gpu[k*P:(k+1)*P], N, P) 
        
        if k>0: 
            
            TT_gpu = gpuarray.GPUArray((N*(k+1)), np.float32) 
            TT_gpu.set(T_gpu[:(k+1)*N])
            Matricetranspose(TT_gpu, T_gpu[:(k+1)*N], N_col=k, N_row=N, TILE_DIM=32)
            MatVectMul(TT_gpu,T_gpu[:(k+1)*N], U_gpu , k,P)
            
            B = T_gpu.copy() 
            MatVectMul(T_gpu, U_gpu, B[k*P:(k+1)*P], N, k) 
            Matvmultiplusv(B[k*N:(k+1)*N], T_gpu[k*N:(k+1)*N], u_scal = -1, N=N, P=1) 
    
        norme_gpu = gpuarray.GPUArray((1), np.float32)
        

        Lambda[k] = Norme2_2(T[k*N:(k+1)*N], norme_gpu)
        Matvmultiplusv(T[k*N:(k+1)*N],u_scal=1.0/Lambda[k],N=N, P=1)
        if abs(Lambda[k] - mu) < epsilon*Lambda[k]: 
            break
        mu = Lambda[k]
    
    Y_gpu = gpuarray.GPUArray((N,P), np.float32)
    MatVectMul(T_gpu[k*N:(k+1)*N], PT_gpu[k*nbr_composantes(k+1)*nbr_composantes],Y, N=N, P=1)
    Matvmultiplusv(R_gpu, Y_gpu, v_scal=-Lambda[k], N_COL=P, N_ROW=N)
            

    for k in range(nbr_composantes):
    Matvmultiplusv(T_gpu[k*N:(k+1)*N], u_scal= Lambda[k], N_COL=1, N_ROW=N)

T= T_gpu.get() 
P = P_gpu.get()

R_gpu.gpudata.free()
X_gpu.gpudata.free()
T_gpu.gpudata.free()
P_gpu.gpudata.free()



