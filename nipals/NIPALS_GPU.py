import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np
from time import time
from kernels.norme_carre import Norme2


class Nipals_GPU():
    def __init__(self, ncomp=None, tol=1e-2, maxiter=1):
        self.tol = tol
        self.maxiter = maxiter
        self.ncomp = ncomp

    # Multiply X.T * th
    def multiply_transpose(self, X_gpu, th_gpu, ph_gpu):
        mult_transpose = SourceModule("""
            #define M %(size_M)d
            #define N %(size_N)d
            
            __global__ void mult_transpose(float *X, float *T ,  float *P)
            {
            
            // Block row and column
            int row = blockIdx.x*blockDim.x + threadIdx.x;
            
            float sum = 0;
            for (int m= 0; m < M; ++m) 
            {
                sum += X[row + m*N]*T[m];
            }
            
            // Write the value to the subvector Tsub
            P[row] = sum;

            }
             """ % {"size_M": self.M, "size_N": self.N})

        mult_transpose_gpu = mult_transpose.get_function("mult_transpose")

        block_size = (min(self.N, 1024), 1, 1)
        grid_size = (int(np.ceil(self.N / block_size[0])), 1, 1)
        mult_transpose_gpu(X_gpu, th_gpu, ph_gpu,
                           block=block_size, grid=grid_size)
        return ph_gpu.get()

    # Normalize ph/ ||ph||
    def normalize_vector(self, ph_gpu):
        normalize = SourceModule("""
            #include <math.h>
            __global__ void normalize(float *ph, float norm2_ph){

            int idx =  blockIdx.x*blockDim.x + threadIdx.x;
            
            ph[idx]=ph[idx]/norm2_ph;                                 
            }
            """)
        normalize_gpu = normalize.get_function("normalize")

        num_threads = int(np.ceil(self.N))
        grid_size = int(np.ceil(num_threads / 1024))

        out = np.zeros(1, dtype=np.float32)
        # print(c)
        out_gpu = gpuarray.to_gpu(out)

        sum_ph = Norme2(ph_gpu, ph_gpu, out_gpu, self.N)

        norm2_ph = np.float32(np.sqrt(sum_ph))

        if grid_size > 1:
            block_size = 1024
        else:
            block_size = num_threads

        normalize_gpu(ph_gpu, norm2_ph, block=(
            block_size, 1, 1), grid=(grid_size, 1, 1))

        return ph_gpu.get()

    # Multiply X * ph
    #th_old_gpu = th_gpu.copy()
    def multipy(self, X_gpu, ph_gpu, th_gpu):
        mult = SourceModule("""

            #include <math.h>
            #define N %(size)d

            
            __global__ void mult(float *X, float *P, float *T)
            {
            
            // Block row and column
            int row = blockIdx.x*blockDim.x + threadIdx.x;
            
            float sum = 0;
            for (int n = 0; n < N; ++n) 
            {
                sum += X[row*N + n]*P[n];
            }
            
            // Write the value to the subvector Tsub
            T[row] = sum;
        }
        """ % {"size": self.N})

        mult_gpu = mult.get_function("mult")
        block_size = (min(self.M, 1024), 1, 1)
        grid_size = (int(np.ceil(self.M / block_size[0])), 1, 1)
        mult_gpu(X_gpu, ph_gpu, th_gpu, block=block_size, grid=grid_size)
        return th_gpu.get()

    # Multiply X - th @ ph.T  ( Mx1 * 1xN)
    def update(self, X_gpu, th_gpu, ph_gpu):
        outer_mult = SourceModule("""
            #include <stdio.h>
            
            # define M %(size_M)d
            # define N %(size_N)d

            __global__ void outer_mult(float *X, float *T ,  float *P)
            {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // Block row and column
            int row = by*blockDim.y + ty;
            int col = bx*blockDim.x + tx;

            //int dim = gridDim.x*blockDim.x;
            if (row < M && col < N){
                int idx = row*N + col ;
                X[idx] -= T[row]*P[col];
            }

            }
            """ % {"size_M": self.M, "size_N": self.N})

        outer = outer_mult.get_function("outer_mult")
        # Maybe modify this because gridDim in x direc is big !
        block_size = (min(self.N, 32), min(self.M, 32), 1)
        grid_size = (
            int(np.ceil(self.N / block_size[0])), int(np.ceil(self.M / block_size[1])), 1)
        outer(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
        return X_gpu

    # Compute eigenvalue
    def get_eigenvalue(self,th_gpu):
        out = np.zeros(1, dtype=np.float32)
        # print(c)
        out_gpu = gpuarray.to_gpu(out)
        sum_th  = Norme2(th_gpu, th_gpu, out_gpu, self.M)
        norm2_th = np.sqrt(sum_th)
        return norm2_th

    @staticmethod
    def normalize(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        X = X - X_mean
        X = X / X_std
        # Normally data is in NxM but we need to return the transpose
        return X.T

    def onestepcomp_gpu(self, X_gpu, comp):
        # get comp row
        th = self.X_PCA[:, comp]
        # th = self.X_GPU[:, comp]
        # NOTE : +20s  on test don't get why
        ph = np.zeros((self.N,)).astype(np.float32)
        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)
        eig = 0

        for j in range(self.maxiter):
            t1 = time()
            # Normalize X.T*th
            self.multiply_transpose(X_gpu, th_gpu, ph_gpu)
            t2 = time()
            print('Time for mult_transpose', t2-t1)

            # Normalize ph/ ||ph||
            t1 = time()
            self.normalize_vector(ph_gpu)
            t2 = time()
            print('Time for normalize_vector', t2-t1)

            # Multiply X * ph
            t1 = time()
            self.multipy(X_gpu, ph_gpu, th_gpu)
            t2 = time()
            print('Time for multipy  X * ph ', t2-t1)

            # Compute eigenvalue
            t1 = time()
            eigh = self.get_eigenvalue(th_gpu)
            t2 = time()
            print('Time for multipy eigenvalue', t2-t1)

            if(np.abs(eigh - eig) < self.tol):
                break
            eig = eigh
        return th_gpu, ph_gpu, eigh

    def fit_on_GPU(self, X):
        """
        fit method
        -------
        parametres 

        output
        ------
        """
        self.X = X.astype(np.float32)

        # move to GPU ,
        # TODO : normlize on GPU
        self.normalized_X = self.normalize(self.X)
        self.X_PCA = self.normalized_X
        self.X_GPU = gpuarray.to_gpu(self.X_PCA)

        # should correspond to X.T.shape
        self.M, self.N = self.X_GPU.shape

        if self.ncomp is None:
            ncomp = min(self.X.shape)
        else:
            try:
                assert self.ncomp <= min(
                    self.M, self.N), "can't have this value will set ncomp to{}".format(min(X.shape))
                ncomp = self.ncomp
            except AssertionError as msg:
                print(msg)
                ncomp = min(self.X.shape)

        eig = np.empty((ncomp,)).astype(np.float32)
        loadings = np.empty((self.N, ncomp)).astype(np.float32)
        scores = np.empty((self.M, ncomp)).astype(np.float32)

        # initialize outputs on gpu
        self.loadings_gpu = gpuarray.to_gpu(loadings)
        self.scores_gpu = gpuarray.to_gpu(scores)

        for comp in range(ncomp):
            # Calculate on full matrix
            th_gpu, ph_gpu, eigh = self.onestepcomp_gpu(self.X_GPU, comp)

            # Update X
            self.X_GPU = self.update(self.X_GPU, th_gpu, ph_gpu)
            self.loadings_gpu[:, comp] = ph_gpu.get()
            self.scores_gpu[:, comp] = th_gpu.get()
            eig[comp] = eigh

        # Get results
        self.eig = eig
        # self.scores = scores
        # self.loadings = loadings

        return True

    def transform_gpu(self):
        # Rertrieve the eigenvectors from score T= US where S is diag
        self.eig_gpu = gpuarray.to_gpu(self.eig)

        U = self.scores_gpu.shape / self.eig

        Z = U.T @ self.normalized_X
        return Z.T

# X = np.random.randn(100, 100).astype(np.float32)

# nips = Nipals_GPU()

# nips.fit_on_GPU(X)

# print(X)
