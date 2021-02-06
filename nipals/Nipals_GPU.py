import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np


# Multiply X.T * th
mult_transpose = SourceModule("""
     #define M %(size_M)d
     #define N %(size_N)d
    
    __global__ void mult_transpose(float *X, float *T ,  float *P)
    {
    
    // Block row and column
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    float sum = 0;
    for (int m= 0; m < M; ++m) 
    {
        sum += X[row + m*N]*T[m];
    }
     
     // Write the value to the subvector Tsub
     P[row] = sum;

    }
    """ % {"size_M": M, "size_N": N})


mult_transpose_gpu = mult_transpose.get_function("mult_transpose")


def multiply_transpose(X_gpu, th_gpu, ph_gpu):

    block_size = (min(N, 1024), 1, 1)
    grid_size = (int(np.ceil(N / block_size[0])), 1, 1)

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

    #include <math.h>
    #define N %(size)d

    
    __global__ void mult(float *X, float *P, float *T)
    {
    
    // Block row and column
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    float sum = 0;
    for (int n = 0; n < N; ++n) 
    {
        sum += X[row*N + n]*P[n];
    }
     
     // Write the value to the subvector Tsub
     T[row] = sum;
}
""" % {"size": N})

mult_gpu = mult.get_function("mult")


def multipy(X_gpu, ph_gpu, th_gpu):
    block_size = (min(M, 1024), 1, 1)
    grid_size = (int(np.ceil(M / block_size[0])), 1, 1)
    mult_gpu(X_gpu, ph_gpu, th_gpu, block=block_size, grid=grid_size)
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




class Nipals_GPU():
    def __init__(self, ncomp=None, tol=1e-2, maxiter=100):
        self.tol = tol
        self.maxiter = maxiter
        self.ncomp = ncomp
    
    @staticmethod
    def normalize(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        X = X - X_mean
        X = X / X_std
        return X.T

    def onestepcomp_gpu(self, X_gpu, comp,N):
        # get comp row
        th = X[:, comp]
        ph = np.zeros((N,)).astype(np.float32)
        eigh = np.zeros((1,)).astype(np.float32)

        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)
        eigh_gpu = gpuarray.to_gpu(eigh)
        eig = 0

        for j in range(max_iter):
            ph = multiply_transpose(X_gpu, th_gpu, ph_gpu)
            # Normalize ph/ ||ph||
            ph = normalize_vector(ph_gpu)
            # Multiply X * ph
            th = multipy(X_gpu, ph_gpu, th_gpu)
            # Compute eigenvalue
            eigh = get_eigenvalue(th_gpu, eigh_gpu)
            if(np.abs(eigh - eig) < tol):
                break
            eig = eigh
        
        print('GPU eigenvalue', eigh)
        return th, ph, eigh

    def fit_on_GPU(self, X):
        """
        fit method
        -------
        parametres 

        output
        ------
        """
        M, N = X.shape
        self.X = X.astype('float')

        # mov to GPU , 
        # TODO : on GPU
        self.normalized_X = self.normalize(self.X)

        self.X_PCA = self.normalized_X
        # print('X input shape (N x M) :', self.X.shape)
        # print('X_pca shape (M x N) :', self.X_PCA.shape)
        self.X_GPU = gpuarray.to_gpu(self.X_PCA)

        nr, nc = self.X_GP.shape
        if self.ncomp is None:
            ncomp = min(self.X.shape)
        else:
            try:
                assert self.ncomp <= min(
                    nr, nc), "can't will set this to{}".format(min(X.shape))
                ncomp = self.ncomp
            except AssertionError as msg:
                print(msg)
                ncomp = min(self.X.shape)

        # initialize outputs
        eig = np.empty((ncomp,))
        loadings = np.empty((nc, ncomp))
        scores = np.empty((nr, ncomp))

        for comp in range(ncomp):
            # Calculate on full matrix
            th, ph, eigh = self.onestepcomp(self.X_PCA, comp)
            # Update X
            self.X_PCA = self.X_PCA - np.outer(th, ph)
            loadings[:, comp] = ph
            scores[:, comp] = th
            eig[comp] = eigh

        # Finalize eigenvalues and subtract from scores
        self.eig = pd.Series(eig)
        # Convert results to DataFrames
        # pd.DataFrame(scores, index=self.X.index, columns=["PC{}".format(i+1) for i in range(ncomp)])
        self.scores = scores
        # pd.DataFrame(loadings, index=self.X.columns, columns=["PC{}".format(i+1) for i in range(ncomp)])
        self.loadings = loadings

        return True
