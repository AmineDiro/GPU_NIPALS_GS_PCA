import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np


M = 5
N = 3

# Allocate memory for the Kth component
X = np.random.randn(M, N).astype(np.float32)
th = np.random.randn(M).astype(np.float32)
ph = np.zeros((N,)).astype(np.float32)
eigh = np.zeros((1,)).astype(np.float32)

X_gpu = gpuarray.to_gpu(X)
th_gpu = gpuarray.to_gpu(th)
ph_gpu = gpuarray.to_gpu(ph)
eigh_gpu = gpuarray.to_gpu(eigh)


# Multiply X.T * th
mod = SourceModule("""
    #define N 5
    __global__ void mult_transpose(float *X, float *th, float *ph){
        int a_idx = blockIdx.x + threadIdx.x * blockDim.y;
        
        __shared__ float temp[5]; 
        temp[threadIdx.x] =  X[a_idx] * th[threadIdx.x];
         
        __syncthreads();
        
         if (0 == threadIdx.x){ 
            float sum = 0;
            for (int i = 0; i < N; i++){
                sum += temp[i];
            }
            ph[blockIdx.x]= sum;
        }

    }
    """)

prog = mod.get_function("mult_transpose")

prog(X_gpu, th_gpu, ph_gpu, block=(M, N, 1), grid=(N, 1, 1))


print(ph_gpu)

mod = SourceModule("""
    #include <math.h>
    #define N 3
    
    __global__ void normalize(float *ph){
    
    __shared__ float temp[N];
    
    if(0== blockIdx.x){
        temp[threadIdx.x]= ph[threadIdx.x]*ph[threadIdx.x];
    }    
    
    __syncthreads();
    
    if (0 == threadIdx.x){ 
        float sum = 0;
        for (int i = 0; i < N; i++){
            sum += temp[i];            
        }
        ph[threadIdx.y]=ph[threadIdx.y]/sqrt(sum);   
    }                     
     
    }

    """)

prog = mod.get_function("normalize")
prog(ph_gpu, block=(N, N, 1), grid=(1, 1, 1))


# Multiply X * ph

th_old_gpu = th_gpu.copy()

mod = SourceModule("""
    #define N 3
    __global__ void mult(float *X,float *ph, float *th){                
        int idx = threadIdx.x + blockDim.x*blockIdx.x;
        
         __shared__ float temp[N]; 
        temp[threadIdx.x] =  X[idx] * ph[threadIdx.x];
         
        __syncthreads();
        
         if (0 == threadIdx.x){ 
            float sum = 0;
            for (int i = 0; i < N; i++){
                sum += temp[i];
            }
            th[blockIdx.x]= sum;
        }

    }
    """)


prog = mod.get_function("mult")

prog(X_gpu, ph_gpu, th_gpu, block=(N, 1, 1), grid=(M, 1, 1))

# Compute eigenvalue
mod = SourceModule("""
     #include <math.h>
     #define M %(size)d
        __global__ void norm2(float *th,float *eigh){                
            
        __shared__ float temp[M]; 
        
        temp[threadIdx.x] =  th[threadIdx.x] * th[threadIdx.x];
         
        __syncthreads();
        
         if (0 == threadIdx.x){ 
         
            float sum = 0;
            for (int i = 0; i < M; i++){
                sum += temp[i];
            }
            eigh[0] = sqrt(sum);
        }

    }
    """ % {"size": M})

prog = mod.get_function("norm2")

prog(th_gpu, eigh_gpu, block=(M, 1, 1), grid=(1, 1, 1))
