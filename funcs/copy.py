import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np

copy = SourceModule("""
            __global__ void copy(float *outMat, float *inMat)
            {
            int TILE_DIM = 32; 
            int x = blockIdx.x * TILE_DIM + threadIdx.x;
            int y = blockIdx.y * TILE_DIM + threadIdx.y;
            int width = gridDim.x * TILE_DIM;

            for (int k = 0 ; k < TILE_DIM ; k += BLOCK_ROWS)
                outMat[(y+k)*width + x] = inMat[(y+k)*width + x];
            }
            """)

THREADS_PER_BLOCK = 1024

copy_gpu = copy.get_function("copy")

a = np.random.randint(low=-10, high=10, size=N_COL, dtype=np.int32)
a = np.random.randint(low=-10, high=10, size=N_COL, dtype=np.int32)
a = a.astype(np.float32)
# print(a)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

c = np.zeros(1, dtype=np.float32)
# print(c)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(c_gpu, c)


Norme2_2(a_gpu, a_gpu, c_gpu)

cuda.memcpy_dtoh(c, c_gpu)

c_cpu = np.dot(a, a)

print(c_cpu)

print(np.allclose(c_cpu, c))
