import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler
import pycuda.gpuarray as gpuarray

import numpy as np


ncol=1000
nrow=1000
tile_dim=32
def Matricetranspose(out_gpu, inp_gpu, N_COL=ncol, N_ROW=nrow, TILE_DIM=tile_dim):
        kernel_code_template = """

        
            __global__ void transpose(float * output, float * input){
                __shared__  float tile[%(TILE_DIM)s][%(TILE_DIM)s+1];

                    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
                    int index_y = threadIdx.y + %(TILE_DIM)s   * blockIdx.y;

               
                    for (int j = 0; j < %(TILE_DIM)s; j += blockDim.y) {
                        int index_y_ = index_y+j;
                        if (index_x<%(N_COL)s && index_y_<%(N_ROW)s)
                            
                          tile[threadIdx.y + j][threadIdx.x] = input[index_y_ * %(N_COL)s + index_x];
                    }
                    __syncthreads();

                    index_x = threadIdx.x + blockDim.x * blockIdx.y;
                    index_y = threadIdx.y + %(TILE_DIM)s   * blockIdx.x;

                
                    for (unsigned j = 0; j < %(TILE_DIM)s; j += blockDim.y) {
                        int index_y_ = index_y+j;
                        if (index_x<%(N_ROW)s && index_y_<%(N_COL)s)
                            
                            output[index_y_ * %(N_COL)s + index_x] = tile[threadIdx.x][threadIdx.y + j];
                    }
            }

            """

        kernel_code = kernel_code_template % {
                    'N_COL': ncol,
                    'N_ROW': nrow  ,
                    'TILE_DIM':tile_dim
            }
        mod = compiler.SourceModule(kernel_code)
        
        
        func = mod.get_function("transpose")

        blocksPerGrid = (int(np.ceil(float(ncol)/float(32))),int(np.ceil(float(nrow)/float(32))),1)
        func(res_gpu, inp_gpu,block=(32,32,1), grid=blocksPerGrid)



M_mat = np.reshape(np.random.randint(-5,5,size=ncol*nrow), (nrow,nrow))
M=M_mat.flatten()
M=M_mat.astype(np.float32)
M_gpu = cuda.mem_alloc(M.nbytes)
cuda.memcpy_htod(M_gpu,M)

inp_gpu = gpuarray.GPUArray((nrow*ncol), np.float32)
inp_gpu.set(M.astype(np.float32))

res_gpu = gpuarray.GPUArray((ncol*ncol), np.float32)


res = np.zeros(ncol*nrow, dtype= np.float32)
res_gpu = cuda.mem_alloc(res.nbytes)
cuda.memcpy_htod(res_gpu, res)


blocksPerGrid = (int(np.ceil(float(nrow)/float(64))),int(np.ceil(float(ncol)/float(1024))),1)
Matricetranspose(res_gpu,M_gpu,ncol,nrow)

cuda.memcpy_dtoh(res, res_gpu)
print(res)
print((np.transpose(M_mat)).flatten())
print(np.allclose((np.transpose(M_mat)).flatten(),res))
