import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np

# GLOBAL VARIABLES
K = 2
M = 5
N = 3

# Multiply X.T * th
mult_transpose = SourceModule("""
    #define N %(size)d

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
    """ % {"size": M})

mult_transpose_gpu = mult_transpose.get_function("mult_transpose")


def multiply_transpose(X_gpu, th_gpu, ph_gpu):
    # TODO : define GridSize with some smart shit and reduce the Block size to 32x32
    input_size = X_gpu.shape
    mult_transpose_gpu(X_gpu, th_gpu, ph_gpu, block=(M, N, 1), grid=(N, 1, 1))
    return ph_gpu.get()


# Normalize ph/ ||ph||
normalize = SourceModule("""
    #include <math.h>
    #define N %(size)d
    
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

    """ % {"size": N})

normalize_gpu = normalize.get_function("normalize")


def normalize_ph(ph_gpu):
    normalize_gpu(ph_gpu, block=(N, N, 1), grid=(1, 1, 1))
    return ph_gpu.get()


# Multiply X * ph
th_old_gpu = th_gpu.copy()

mult = SourceModule("""
    #define N %(size)d
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
    """ % {"size": N})

mult_gpu = mult.get_function("mult")


def multipy(X_gpu, ph_gpu, th_gpu):
    mult_gpu(X_gpu, ph_gpu, th_gpu, block=(N, 1, 1), grid=(M, 1, 1))
    return th_gpu.get()


# Compute eigenvalue
norm2 = SourceModule("""
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

norm2_gpu = norm2.get_function("norm2")


def get_eigenvalue(th_gpu, eigh_gpu):
    norm2_gpu(th_gpu, eigh_gpu, block=(M, 1, 1), grid=(1, 1, 1))
    return eigh_gpu.get()


if __name__ == '__main__':
    X = np.random.randn(M, N).astype(np.float32)
    th = np.random.randn(M).astype(np.float32)
    ph = np.zeros((N,)).astype(np.float32)
    eigh = np.zeros((1,)).astype(np.float32)

    X_gpu = gpuarray.to_gpu(X)
    th_gpu = gpuarray.to_gpu(th)
    ph_gpu = gpuarray.to_gpu(ph)
    eigh_gpu = gpuarray.to_gpu(eigh)

    ########## One Comp step ##########
    # Multiply X.T * th
    ph = multiply_transpose(X_gpu, th_gpu, ph_gpu)
    # Normalize ph/ ||ph||
    ph = normalize_ph(ph_gpu)
    #Multiply X * ph
    th = multipy(X_gpu, ph_gpu, th_gpu)
    # Compute eigenvalue
    eigh = get_eigenvalue(th_gpu, eigh_gpu)


