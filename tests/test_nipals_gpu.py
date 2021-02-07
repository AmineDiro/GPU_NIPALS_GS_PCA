import sys
import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy.testing as npt
from time import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from nipals.NIPALS_GPU import Nipals_GPU
from nipals.kernels import multiply_transpose, normalize_vector, Norme2, multipy, update

# sys.path.insert(
#    0, '/Users/dihroussi/Google Drive/Documents/ENSAE/GPU/Projet GPU')

N = 100


class TestNipalsGPU(unittest.TestCase):
    def test_mult_transpose(self):

        X = np.random.randn(N, N).astype(np.float32)
        th = np.random.randn(N).astype(np.float32)
        ph = np.zeros((N,)).astype(np.float32)

        X_gpu = gpuarray.to_gpu(X)
        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)

        ph_gpu = multiply_transpose(X_gpu, th_gpu, ph_gpu, N, N)

        npt.assert_allclose(X.T @ th, ph_gpu.get(), rtol=1e-2)

    def test_norm2(self):
        ph = np.random.randn(N).astype(np.float32)
        norm = np.zeros(1, dtype=np.float32)
        ph_gpu = gpuarray.to_gpu(ph)

        norm_gpu = gpuarray.to_gpu(norm)

        norm2 = Norme2(ph_gpu, ph_gpu, norm_gpu, N)

        npt.assert_allclose(np.sum(ph*ph), norm2, rtol=1e-2)

    def test_normalize_vector(self):
        ph = np.random.randn(N).astype(np.float32)

        ph_gpu = gpuarray.to_gpu(ph)

        ph_gpu = normalize_vector(ph_gpu, N)

        npt.assert_allclose(ph / np.sqrt(np.sum(ph*ph)),
                            ph_gpu.get(), rtol=1e-2)

    def test_mult(self):
        X = np.random.randn(N, N).astype(np.float32)
        ph = np.random.randn(N).astype(np.float32)
        th = np.zeros((N,)).astype(np.float32)

        X_gpu = gpuarray.to_gpu(X)
        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)

        th_gpu = multipy(X_gpu, ph_gpu, th_gpu, N, N)

        npt.assert_allclose(X@ph, th_gpu.get(), rtol=1e-2)

    def test_update(self):
        X = np.random.randn(N, N).astype(np.float32)
        ph = np.random.randn(N).astype(np.float32)
        th = np.random.randn(N).astype(np.float32)

        X_gpu = gpuarray.to_gpu(X)
        th_gpu = gpuarray.to_gpu(th)
        ph_gpu = gpuarray.to_gpu(ph)

        dum = X - np.outer(th, ph)

        X_gpu = update(X_gpu, th_gpu, ph_gpu, N, N)

        npt.assert_allclose(dum, X_gpu.get(), rtol=1e-2)

    def test_eig(self):
        N = 10
        X = np.random.randn(N, N)
        n_components = 2
        t1 = time()
        nips = Nipals_GPU(ncomp=n_components)
        assert nips.fit_on_GPU(X)
        t2 = time()
        print('Total time for GPU NIPALS :', t2-t1)
        std = StandardScaler()
        X = std.fit_transform(X)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        npt.assert_allclose(pca.singular_values_, nips.eig, rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
