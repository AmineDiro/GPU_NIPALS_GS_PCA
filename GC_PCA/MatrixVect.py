
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler
import pycuda.gpuarray as gpuarray


#Ce code réalise la multiplication de la matrice M par u_sacl puis additionne le résultat au produit du vecteur V et de v_scal
THREADS_PER_BLOCK = 1024

N = 5# num observations
P = 2 # num variables

def SommeMatrice(M,  V , out, u_scal=1 ,
               v_scal=1 ,  N = N, P=P, THREADS_PER_BLOCK=THREADS_PER_BLOCK):
      
            GRID_SIZE= (P, int((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK))
            
            kernel_code_template = """


                __global__ void VectMat(float *inp, float *vect, float *out)

                {

              __shared__ float temp;
              __shared__ int blockxInd;
              __shared__ int blockyInd;

              temp = %(scalv)s * vect[blockIdx.x];
              blockxInd = %(N)s * blockIdx.x;
              blockyInd = %(THREADS_PER_BLOCK)s * blockIdx.y;

              __syncthreads();

              // 
              
              int idx = threadIdx.x + blockxInd + blockyInd;
              
              if( blockyInd+threadIdx.x  < %(N)s)
                  {
                  out[idx] =  %(scalu)s *inp[idx] + temp;

                }
                }


                """

            kernel_code = kernel_code_template % {
                'scalv': v_scal,
                'scalu': u_scal,
                'N': N,
                'P': P,
                'THREADS_PER_BLOCK': THREADS_PER_BLOCK

            }

            mod = compiler.SourceModule(kernel_code)



            func = mod.get_function("VectMat")
            func(M, V,out, block=(THREADS_PER_BLOCK, 1,1), grid=(GRID_SIZE))




X = np.reshape(np.random.randint( -5, 5 , size=N*P), (N, P))
a = X.T.astype(np.float32)
a = a.flatten()
               
X_gpu = gpuarray.GPUArray((P*N), np.float32)
X_gpu.set(a.astype(np.float32))

#R = np.reshape(range(N*P), (N,P))
R = np.arange(N*P)
R_gpu = gpuarray.GPUArray((N*P), np.float32)
R_gpu.set(R.astype(np.float32))
               
#vector of features sum
V = X.sum(axis=0) # n_variables*1

V_gpu = gpuarray.GPUArray((P,), np.float32)
V_gpu.set(V.astype(np.float32))

scalar= float(1/N)



MatrixSum(X_gpu,v_gpu = V_gpu, out = R_gpu,
               scalar_v= -scalar)


result = R_gpu.get()
result = np.reshape(result, (N,P), order='F')

V = np.repeat([V],N, axis=0)
R_cpu = X-scalar*V

print(np.allclose(result, R_cpu))
