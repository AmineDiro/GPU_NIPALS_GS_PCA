import pycuda.driver as cuda
from pycuda import driver, compiler
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np


# Multiply X.T * th
def multiply_transpose(X_gpu, th_gpu, ph_gpu, M, N):
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
             """ % {"size_M": M, "size_N": N})

    mult_transpose_gpu = mult_transpose.get_function("mult_transpose")

    block_size = (min(N, 1024), 1, 1)
    grid_size = (int(np.ceil(N / block_size[0])), 1, 1)
    mult_transpose_gpu(X_gpu, th_gpu, ph_gpu,
                       block=block_size, grid=grid_size)

    return ph_gpu

# Return norme squared sum(a*a)
def Norme2(a_gpu, b_gpu, c_gpu, N):
    # THREADS_PER_BLOCK = 1024
    kernel_code_template = """

            __global__ void dot(float *a, float *b,float *c) {
          
          __shared__ float temp[1024];
          int index = threadIdx.x + blockIdx.x  *blockDim.x;
          temp[threadIdx.x] = a[index] * b[index];
          
            __shared__ float cache;

          if (threadIdx.x == 0) {
            if ((blockIdx.x + 1) * 1024<= %(N)s)
              cache = 1024;
            else cache = fmodf(%(N)s, 1024);
          }
          __syncthreads();

          if (threadIdx.x == 0) {
              float sum =0;
              for (int i =0; i< cache; i++)
                  sum += temp[i];
              atomicAdd(c, sum);
              }
          
          }
            
         """

    kernel_code = kernel_code_template % {
        'N': N,
    }
    mod = compiler.SourceModule(kernel_code)

    func = mod.get_function("dot")

    blocksPerGrid = (int((N + 1024-1)/1024), 1)

    func(a_gpu, a_gpu, c_gpu, block=(1024, 1, 1), grid=blocksPerGrid)

    return c_gpu.get()

# Normalize ph/ ||ph||
def normalize_vector(ph_gpu, N):
    normalize = SourceModule("""
        #include <math.h>
        __global__ void normalize(float *ph, float norm2_ph){
        int idx =  blockIdx.x*blockDim.x + threadIdx.x;
        
        ph[idx]=ph[idx]/norm2_ph;                                 
        }
        """)
    normalize_gpu = normalize.get_function("normalize")
    num_threads = int(np.ceil(N))
    grid_size = int(np.ceil(num_threads / 1024))
    out = np.zeros(1, dtype=np.float32)
    out_gpu = gpuarray.to_gpu(out)
    # sum_ph = gpuarray.sum(ph_gpu**2)
    sum_ph = Norme2(ph_gpu, ph_gpu, out_gpu, N)
    norm2_ph = np.float32(np.sqrt(sum_ph))
    if grid_size > 1:
        block_size = 1024
    else:
        block_size = num_threads
    normalize_gpu(ph_gpu, norm2_ph, block=(
        block_size, 1, 1), grid=(grid_size, 1, 1))

    return ph_gpu

# Multiply X * ph
def multipy(X_gpu, ph_gpu, th_gpu, M, N):
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
    """ % {"size": N})
    mult_gpu = mult.get_function("mult")
    block_size = (min(M, 1024), 1, 1)
    grid_size = (int(np.ceil(M / block_size[0])), 1, 1)
    mult_gpu(X_gpu, ph_gpu, th_gpu, block=block_size, grid=grid_size)

    return th_gpu


# Multiply X - th @ ph.T  ( Mx1 * 1xN)
def update(X_gpu, th_gpu, ph_gpu, M, N):
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
    outer(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
    return X_gpu

# Use sqrt(norm2)
def get_eigenvalue(th_gpu, M):
    out = np.zeros(1, dtype=np.float32)
    out_gpu = gpuarray.to_gpu(out)
    sum_th = Norme2(th_gpu, th_gpu, out_gpu, M)
    norm2_th = np.sqrt(sum_th)
    return norm2_th
