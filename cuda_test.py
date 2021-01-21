import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import numpy as np


N=1000
BLOCKSIZE = 256


blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (N// BLOCKSIZE +1, 1, 1)


a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N).astype(np.float32)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(c)


mod = SourceModule("""
  __global__ void add(float *a,float *b, float *c, int N)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
      c[idx] = a[idx] + b[idx];
  }
  """)
prog = mod.get_function("add")

prog(a_gpu, b_gpu,c_gpu, np.uint32(N),block=blockDim,grid=gridDim)

print(c_gpu.get())