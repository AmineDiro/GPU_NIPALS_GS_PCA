import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler
import pycuda.gpuarray as gpuarray
import numpy as np



N_col=50
N_row=700
BLOCK_WIDTH =64
THREADS_PER_BLOCK= 1024


def MatVectMul(a_gpu, b_gpu, c_gpu,
               N_COL = N_col, N_ROW = N_row,
                THREADS_PER_BLOCK=THREADS_PER_BLOCK,
               BLOCK_WIDTH = 64
              ):
    
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

          float cSum = (float) 0;
          int threadyInd = blockyInd + threadIdx.x;

          
        
          if (threadyInd < %(N_ROW)s) {

            for (int i=0; i<blockElt; i++)

              cSum += b[i] * a[(blockxInd + i) * (%(N_ROW)s) + (threadyInd)];

            atomicAdd(out + threadyInd, cSum);

          }
        }
            """

        kernel_code = kernel_code_template % {
                 'N_COL' : N_col,
                 'N_ROW' : N_row,
                 'BLOCK_WIDTH' :BLOCK_WIDTH,
                 'THREADS_PER_BLOCK':THREADS_PER_BLOCK

            }
        mod = compiler.SourceModule(kernel_code)
        
        func = mod.get_function("MatvectMulti")
        
        dimGridx = int((N_col+ BLOCK_WIDTH-1)/BLOCK_WIDTH)
        dimGridy= int((N_row + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK)
        blocksPerGrid = (dimGridx,dimGridy)
        
        func(a_gpu, b_gpu, c_gpu, block=(THREADS_PER_BLOCK,1,1), grid=blocksPerGrid)




a_mat = np.reshape(np.random.randint(-5, 5, size=N_col*N_row, dtype= np.int32), (N_row, N_col))

a = np.transpose(a_mat)
a = a.flatten()
a = a.astype(np.float32)

a_gpu = gpuarray.GPUArray((N_col*N_row), np.float32)
a_gpu.set(a.astype(np.float32))

b = np.random.randint(-5, 5, size=N_col, dtype= np.int32)
b = b.astype(np.float32)
b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

c = np.zeros(N_row, dtype= np.float32)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(c_gpu, c)


MatVectMul(a_gpu, b_gpu, c_gpu)

cuda.memcpy_dtoh(c, c_gpu)