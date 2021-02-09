# Test multi

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np


# mult = SourceModule("""

#      # include <math.h>
#      # define N 64


#     __global__ void mult(float *X, float *P, float *T)
#     {

#     // Block row and column
#     int row = blockIdx.x*blockDim.x + threadIdx.x;

#     float sum = 0;
#     for (int n = 0; n < N; ++n)
#     {
#         sum += X[row*N + n]*P[n];
#     }

#      // Write the value to the subvector Tsub
#      T[row] = sum;
# }
# """)


# # tile_size = 32

# # block_size = (min(M, 32), 1, 1)
# # grid_size = (int(np.ceil(M / block_size[0])), 1, 1)


# # X_vid = np.zeros((32,)).astype(np.float32)
# # X_vid_gpu = gpuarray.to_gpu(X_vid)


# # mult_gpu = mult.get_function("MatMulKernel")

# # mult_gpu(X_gpu, ph_gpu, th_gpu, block=block_size, grid=grid_size)


# # X = np.random.randn(M, N).astype(np.float32)
# # # ph = np.random.randn(N).astype(np.float32)
# # th = np.random.randn(M).astype(np.float32)
# # ph = np.zeros((N,)).astype(np.float32)

# # X_gpu = gpuarray.to_gpu(X)
# # th_gpu = gpuarray.to_gpu(th)
# # ph_gpu = gpuarray.to_gpu(ph)


# # Multiply X.T * th
# mult_transpose = SourceModule("""
#      # define M 5
#      # define N  7

#     __global__ void mult_transpose(float *X, float *T ,  float *P)
#     {

#     // Block row and column
#     int row = blockIdx.x*blockDim.x + threadIdx.x;

#     float sum = 0;
#     for (int m= 0; m < M; ++m)
#     {
#         sum += X[row + m*N]*T[m];
#     }

#      // Write the value to the subvector Tsub
#      P[row] = sum;

#     }
#     """ % {"size_M": M, "size_N": N})

# mult_transpose_gpu = mult_transpose.get_function("mult_transpose")


# def multiply_transpose(X_gpu, th_gpu, ph_gpu):

#     block_size = (min(N, 1024), 1, 1)
#     grid_size = (int(np.ceil(N / block_size[0])), 1, 1)
#     print('block_size : ', block_size)
#     print('grid_size : ', grid_size)
#     mult_transpose_gpu(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
#     return ph_gpu.get()


# # ph = multiply_transpose(X_gpu, th_gpu, ph_gpu)


# # np.testing.assert_allclose(X.T@th, ph, rtol=1e-2)

############################################################################################

M = 100
N = 1000
# Multiply th @ ph.T  ( Mx1 * 1xN)


def update(X_gpu, th_gpu, ph_gpu):
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
        """ % {"size_M": M, "size_N": N})

    outer = outer_mult.get_function("outer_mult")
    # Maybe modify this because gridDim in x direc is big !
    block_size = (min(N, 32), min(M, 32), 1)
    grid_size = (
        int(np.ceil(N / block_size[0])), int(np.ceil(M / block_size[1])), 1)
    print('M', M)
    print('N', N)
    print('block_size : ', block_size)
    print('grid_size : ', grid_size)
    outer(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
    return X_gpu.get()


X = np.random.randn(M, N).astype(np.float32)
ph = np.random.randn(N).astype(np.float32)
th = np.random.randn(M).astype(np.float32)


X_gpu = gpuarray.to_gpu(X)
th_gpu = gpuarray.to_gpu(th)
ph_gpu = gpuarray.to_gpu(ph)


X_get = outer_mult(X_gpu, th_gpu, ph_gpu)

test = X - np.outer(th, ph)
np.testing.assert_allclose(test, X_get, rtol=1e-2)
