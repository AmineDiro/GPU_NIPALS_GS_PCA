import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import cumath
import numpy as np
from time import time
# from kernels.norme_carre import Norme2
# from kernels.mult_transpose import mult_transpose
from nipals.kernels import multiply_transpose, normalize_vector, Norme2, multipy, update, get_eigenvalue


class Nipals_GPU():
    def __init__(self, ncomp=None, tol=1e-5, maxiter=500):
        self.tol = tol
        self.maxiter = maxiter
        self.ncomp = ncomp


    @staticmethod
    def normalize(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        X = X - X_mean
        X = X / X_std
        # Normally data is in NxM but we need to return the transpose
        return X.T

    def onestepcomp_gpu(self, X_gpu, comp):
        # get comp row
        th = self.X_PCA[:, comp]
        ph = np.zeros((self.N,)).astype(np.float32)

        # th = self.X_GPU[:, comp]
        # NOTE : +20s  on test don't get why

        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)
        eig = 0

        for j in range(self.maxiter):
            t1 = time()
            # Normalize X.T*th
            multiply_transpose(X_gpu, th_gpu, ph_gpu, self.M, self.N)
            #ph_gpu=  mult_transpose(X_gpu,th_gpu,ph_gpu,self.N)
            t2 = time()
            # print('Time for mult_transpose', t2-t1)

            # Normalize ph/ ||ph||
            t1 = time()
            normalize_vector(ph_gpu, self.N)
            t2 = time()
            # print('Time for normalize_vector', t2-t1)

            # Multiply X * ph
            t1 = time()
            multipy(X_gpu, ph_gpu, th_gpu, self.M, self.N)
            t2 = time()
            # print('Time for multipy  X * ph ', t2-t1)

            # Compute eigenvalue
            t1 = time()
            eigh = get_eigenvalue(th_gpu,self.M)
            t2 = time()
            # print('Time for multipy eigenvalue', t2-t1)

            if(np.abs(eigh - eig) < self.tol):
                break
            eig = eigh
        return th_gpu, ph_gpu, eigh

    def fit_on_GPU(self, X):
        """
        fit method
        -------
        parametres : X data N x N Matrix 

        output : True       
        ------
        """
        self.X = X.astype(np.float32)

        # move to GPU ,
        # TODO : normlize on GPU
        self.normalized_X = self.normalize(self.X)
        self.X_PCA = self.normalized_X
        self.X_GPU = gpuarray.to_gpu(self.X_PCA)

        # should correspond to X.T.shape
        self.M, self.N = self.X_GPU.shape

        if self.ncomp is None:
            ncomp = min(self.X.shape)
        else:
            try:
                assert self.ncomp <= min(
                    self.M, self.N), "can't have this value will set ncomp to{}".format(min(X.shape))
                ncomp = self.ncomp
            except AssertionError as msg:
                print(msg)
                ncomp = min(self.X.shape)

        eig = np.empty((ncomp,)).astype(np.float32)
        loadings = np.empty((self.N, ncomp)).astype(np.float32)
        scores = np.empty((self.M, ncomp)).astype(np.float32)

        # initialize outputs on gpu
        self.loadings_gpu = gpuarray.to_gpu(loadings)
        self.scores_gpu = gpuarray.to_gpu(scores)

        for comp in range(ncomp):
            # Calculate on full matrix
            th_gpu, ph_gpu, eigh = self.onestepcomp_gpu(self.X_GPU, comp)

            # Update X
            self.X_GPU = update(self.X_GPU, th_gpu, ph_gpu,self.M,self.N)
            self.loadings_gpu[:, comp] = ph_gpu.get()
            self.scores_gpu[:, comp] = th_gpu.get()
            eig[comp] = eigh

        # Get results
        self.eig = eig
        # self.scores = scores
        # self.loadings = loadings

        return True

    # def transform_gpu(self):
    #     # Rertrieve the eigenvectors from score T= US where S is diag
    #     self.eig_gpu = gpuarray.to_gpu(self.eig)

    #     U = self.scores_gpu.shape / self.eig

    #     Z = U.T @ self.normalized_X
    #     return Z.T

class Nipals_CPU():

    def __init__(self, ncomp=None, tol=0.00001, maxiter=500):
        self.tol = tol
        self.maxiter = maxiter
        self.ncomp = ncomp

    # TODO : Add class methods bach ndiro l'import nichane
    # Write class method bach normalizé lblane

    # TODO Fix this
    @staticmethod
    def normalize(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        X = X - X_mean
        X = X / X_std
        return X.T

    def onestepcomp(self, X, comp):
        # TODO : Check for missing values
        # Choose the column of X with highest variance
        xvar = np.nanvar(X, axis=0, ddof=1)
        #startcol_use = np.where(xvar == xvar.max())[0][0]
        #th = X[:, startcol_use]

        th = X[:, comp]

        it = 0
        for j in range(self.maxiter):
            # loadings
            ph = X.T.dot(th)
            # Normalize
            ph = ph / np.sqrt(np.sum(ph*ph))
            # Scores update
            th_old = th
            th = X.dot(ph)

            # Check convergence
            if np.sum((th-th_old)**2) < self.tol:
                break
            
        eigh = np.sqrt(np.sum(th*th))

        return th, ph, eigh

    def fit(self, X, startcol=None):
        """
        fit method
        -------
        parametres 

        output
        ------
        """
        nr, nc = X.shape
        self.X = X.astype('float')

        # Save for  later
        self.normalized_X = self.normalize(self.X)

        self.X_PCA = self.normalized_X
        # print('X input shape (N x M) :', self.X.shape)
        # print('X_pca shape (M x N) :', self.X_PCA.shape)

        nr, nc = self.X_PCA.shape
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
        self.eig = eig
        # Convert results to DataFrames
        self.scores = scores
        self.loadings = loadings

        return True

    def transform(self):
        # Rertrieve the eigenvectors from score T= US where S is diag
        U = self.scores / self.eig
        print('scores shape:', self.scores.shape)
        print('eig shape:', self.eig)
        print('U shape:', U.shape)
        Z = U.T @ self.normalized_X
        return Z.T

    def fit_transform(self, X, startcol=None):
        """
        fit_ transform method
        -------
        parametres 

        output
        ------
        """
        self.fit(X, startcol=startcol)
        return self.transform()
