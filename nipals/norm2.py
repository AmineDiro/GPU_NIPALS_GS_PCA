import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler
import pycuda.gpuarray as gpuarray

import numpy as np


N_COL = 100000
N_ROW = 1


THREADS_PER_BLOCK = 1024


def Norme2_2(a_gpu, b_gpu, c_gpu, N_COL=N_COL, N_ROW=N_ROW):
    THREADS_PER_BLOCK = 1024
    kernel_code_template = """

            __global__ void dot(float *a, float *b,float *c) {
          
          __shared__ float temp[1024];
          int index = threadIdx.x + blockIdx.x  *blockDim.x;
          temp[threadIdx.x] = a[index] * b[index];
          
            __shared__ int cache;

          if (threadIdx.x == 0) {
            if ((blockIdx.x + 1) * 1024<= %(N_COL)s)
              cache = 1024;
            else cache = fmodf(%(N_COL)s, 1024);
          }
          __syncthreads();

          if (threadIdx.x == 0) {
              int sum =0;
              for (int i =0; i< cache; i++)
                  sum += temp[i];
              atomicAdd(c, sum);
              }
          
          }
            
         """

    kernel_code = kernel_code_template % {
        'N_COL': N_COL,
        'N_ROW': N_ROW

    }
    mod = compiler.SourceModule(kernel_code)

    func = mod.get_function("dot")

    blocksPerGrid = (int((N_COL + 1024-1)/1024), 1)

    func(a_gpu, a_gpu, c_gpu, block=(1024, 1, 1), grid=blocksPerGrid)


# a = np.random.randint(low=-10, high=10, size=N_COL, dtype=np.int32)
# a = a.astype(np.float32)
# # print(a)
# a_gpu = cuda.mem_alloc(a.nbytes)
# cuda.memcpy_htod(a_gpu, a)

# c = np.zeros(1, dtype=np.float32)
# # print(c)
# c_gpu = cuda.mem_alloc(c.nbytes)
# cuda.memcpy_htod(c_gpu, c)


# Norme2_2(a_gpu, a_gpu, c_gpu)

# cuda.memcpy_dtoh(c, c_gpu)

# c_cpu = np.dot(a, a)

# print(c_cpu)

# print(np.allclose(c_cpu, c))
