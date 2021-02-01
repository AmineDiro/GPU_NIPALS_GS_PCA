import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np

# GLOBAL VARIABLES
K = 2
M = 10
N = 10

# Multiply X.T * th
mult_transpose = SourceModule("""
    #define M %(size)d
    __global__ void mult_transpose(float *X, float *th, float *ph){
        int a_idx = blockIdx.x + threadIdx.x * gridDim.x;
        
        __shared__ float temp[M]; 
        temp[threadIdx.x] =  X[a_idx] * th[threadIdx.x];
         
        __syncthreads();
        
         if (0 == threadIdx.x){ 
            float sum = 0;
            for (int i = 0; i < M; i++){
                sum += temp[i];
            }
            ph[blockIdx.x]= sum;
        }

    }
    """ % {"size": M})

mult_transpose_gpu = mult_transpose.get_function("mult_transpose")


def multiply_transpose(X_gpu, th_gpu, ph_gpu):
    # TODO : define GridSize with some smart shit and reduce the Block size to 32x32
    if M < 1024:
        block_size = (M, 1, 1)
        grid_size = (N, 1, 1)
    else:
        pass
        # Raise error
    mult_transpose_gpu(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
    return ph_gpu.get()


# Normalize ph/ ||ph||
normalize = SourceModule("""
    #include <math.h>
    __global__ void normalize(float *ph, float norm2_ph){

    int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    
    ph[idx]=ph[idx]/norm2_ph;                                 
    }
    """)
normalize_gpu = normalize.get_function("normalize")


def normalize_vector(ph_gpu):
    num_threads = int(np.ceil(N))
    grid_size = int(np.ceil(num_threads / 1024))

    # NOTE : Tried to implement with from sirst princples GPU but a lot  problems
    # Check out : https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
    sum_ph = gpuarray.sum(ph_gpu**2)
    norm2_ph = np.float32(np.sqrt(sum_ph.get()))

    if grid_size > 1:
        block_size = 1024
    else:
        block_size = num_threads

    normalize_gpu(ph_gpu, norm2_ph, block=(
        block_size, 1, 1), grid=(grid_size, 1, 1))
    return ph_gpu.get()


# Multiply X * ph
#th_old_gpu = th_gpu.copy()
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
def get_eigenvalue(th_gpu, eigh_gpu):
    sum_th = gpuarray.sum(th_gpu**2)
    norm2_th = np.sqrt(sum_th.get())
    return norm2_th


def onecomp_cpu(X, th, ph):
    # loadings
    ph_cpu = X.T.dot(th)
    # Normalize
    ph_cpu = ph_cpu / np.sqrt(np.sum(ph_cpu*ph_cpu))
    # Scores update
    th_cpu = X.dot(ph_cpu)
    eig_cpu = np.sqrt(np.sum(th_cpu*th_cpu))
    return eig_cpu, th_cpu, ph_cpu


if __name__ == '__main__':
    X = np.random.randn(M, N).astype(np.float32)
    # Initiliaze with the kth column of X
    th = X[:, K]
    ph = np.zeros((N,)).astype(np.float32)
    eigh = np.zeros((1,)).astype(np.float32)

    X_gpu = gpuarray.to_gpu(X)
    th_gpu = gpuarray.to_gpu(th)
    ph_gpu = gpuarray.to_gpu(ph)
    eigh_gpu = gpuarray.to_gpu(eigh)

    ########## One Comp step GPU ##########
    eig_cpu, th_cpu, ph_cpu = onecomp_cpu(X, th, ph)
    print('CPU eigenvalue', eig_cpu)

    ########## One Comp step GPU ##########
    # Multiply X.T * th
    ph = multiply_transpose(X_gpu, th_gpu, ph_gpu)

    # Normalize ph/ ||ph||
    ph = normalize_vector(ph_gpu)

    # Multiply X * ph
    th = multipy(X_gpu, ph_gpu, th_gpu)

    # Compute eigenvalue
    eigh = get_eigenvalue(th_gpu, eigh_gpu)

    print('GPU eigenvalue', eigh)
