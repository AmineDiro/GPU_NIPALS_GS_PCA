import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler
import pycuda.gpuarray as gpuarray
import numpy as np


def mult_transpose(a_gpu, b_gpu, c_gpu, N, THREADS_PER_BLOCK=1024, BLOCK_WIDTH=64):
    kernel_code_template = """
            __global__ void MatvectMulti(float *a, float *in,float *out ) {
              
          __shared__ int blockElt;
          __shared__ int blockxInd;
          __shared__ int blockyInd;

          
          if (threadIdx.x == 0) {
            if ((blockIdx.x + 1) * %(BLOCK_WIDTH)s <= %(N_COL)s)
              blockElt = %(BLOCK_WIDTH)s;
  
            else blockElt = fmodf(%(N_COL)s, %(BLOCK_WIDTH)s);
            blockxInd = blockIdx.x * %(BLOCK_WIDTH)s;
            blockyInd = blockIdx.y * %(THREADS_PER_BLOCK)s;
          }

          __syncthreads();

          
          __shared__ float b[%(BLOCK_WIDTH)s];

          if (threadIdx.x < blockElt) 
            b[threadIdx.x] = in[blockxInd + threadIdx.x];

          __syncthreads();

          float cSum = 0;
          int threadyInd = blockyInd + threadIdx.x;

          
        
          if (threadyInd < %(N_ROW)s) {

            for (int i=0; i<blockElt; i++)

              cSum += b[i] * a[(blockxInd + i) * (%(N_ROW)s) + (threadyInd)];

            atomicAdd(out + threadyInd, cSum);

          }
        }
            """

    kernel_code = kernel_code_template % {
        'N_COL': N,
        'N_ROW': N,
        'BLOCK_WIDTH': BLOCK_WIDTH,
        'THREADS_PER_BLOCK': THREADS_PER_BLOCK

    }
    mod = compiler.SourceModule(kernel_code)

    func = mod.get_function("MatvectMulti")

    dimGridx = int((N + BLOCK_WIDTH-1)/BLOCK_WIDTH)
    dimGridy = int((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
    blocksPerGrid = (dimGridx, dimGridy)

    func(a_gpu, b_gpu, c_gpu, block=(THREADS_PER_BLOCK, 1, 1), grid=blocksPerGrid)

    return c_gpu


# N = 10

# BLOCK_WIDTH = 64
# THREADS_PER_BLOCK = 1024


# a = np.random.randn(N, N).astype(np.float32)
# b = np.random.randn(N).astype(np.float32)
# c = np.zeros(N, dtype=np.float32)

# a_gpu = gpuarray.to_gpu(a)
# b_gpu = gpuarray.to_gpu(b)
# c_gpu = gpuarray.to_gpu(c)


# c_gpu = mult_transpose(a_gpu, b_gpu, c_gpu, N)

# # cuda.memcpy_dtoh(c, c_gpu)

# print(c_gpu.get())

# print(a.T@b)
