# Test multi

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np

mult = SourceModule("""

     #include <math.h>
     #define BLOCK_SIZE 32

    __device__ float* GetSubVector(float *V, int row)
    {
        return &V[BLOCK_SIZE * row]; 
        
    }

    __device__ float* GetSubMatrix(float *X, int row, int col)
    {
        // First element of the block                 
        return &X[BLOCK_SIZE * row + BLOCK_SIZE * col]; 
    }
       
    __device__ float GetMatrixElement(float* X, int row, int col, int N , int blockCol )
    {
        return X[N * row + col - BLOCK_SIZE * blockCol]; 

        }
    
   
    __global__ void MatMulKernel(float *X, float *P, float *T, int N, int M)
    {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Vecteur result 
    float *Tsub = GetSubVector(T, blockIdx.x);

    // Each thread gives one element of  Tsub
    // Then adds it to Tvalue    
    float Tvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of X and subvectors of P
    // Required to compute the vector Tsub
    // Multiply each pair of sub-matrices together
    // and accumulate the results

    for (int m = 0; m < gridDim.y; ++m) 
    {
        // Get sub-matrix Xsub of X
        float* Xsub = GetSubMatrix(X, blockIdx.x, m);
                
        // Get subvector  Psub of P
        float* Psub = GetSubVector(P, m);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float X_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float P_shared[BLOCK_SIZE];

        // Load Xsub and Vsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        X_shared[threadIdx.x][threadIdx.y] = GetMatrixElement(Xsub, threadIdx.x, threadIdx.y, N , blockIdx.y);
        P_shared[threadIdx.x] = Psub[threadIdx.x];
        
        // Synch threads !! 
        __syncthreads();

         // TODO : add condition to not go past
         // maybe use threadIdx.y ?? 
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Tvalue +=  X_shared[threadIdx.x][e] * P_shared[e];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of X and V in the next iteration
        __syncthreads();
    }
     
     // Write the value to the subvector Tsub
     Tsub[threadIdx.x] = Tvalue;
}
""")


M =  64
N = 64


X = np.random.randn(M, N).astype(np.float32)
ph = np.random.randn(N).astype(np.float32)
th = np.zeros((M,)).astype(np.float32)

X_gpu = gpuarray.to_gpu(X)
th_gpu = gpuarray.to_gpu(th)
ph_gpu = gpuarray.to_gpu(ph)

tile_size = 32

block_size = (min(N, 32), min(M, 32), 1)
grid_size = (int(np.ceil(N / block_size[0])),
             int(np.ceil(M / block_size[1])), 1)


X_vid = np.zeros((32,)).astype(np.float32)
X_vid_gpu = gpuarray.to_gpu(X_vid)


mult_gpu = mult.get_function("MatMulKernel")

mult_gpu(X_gpu, ph_gpu, th_gpu, np.int32(N), np.int32(
    M), block=block_size, grid=grid_size)


np.testing.assert_allclose(X@ph, th_gpu.get(), rtol=1e-2)
