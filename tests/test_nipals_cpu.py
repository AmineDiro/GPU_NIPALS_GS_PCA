from nipals.NIPALS import Nipals_CPU
import sys
import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy.testing as npt
from time import time

# sys.path.insert(
#    0, '/Users/dihroussi/Google Drive/Documents/ENSAE/GPU/Projet GPU')


class TestNipalsCPU(unittest.TestCase):

    def test_pca(self):
        # generate data
        n_components = 2

        X = np.random.randn(100, 30)

        # rng = np.random.RandomState(1)
        # X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

        nips = Nipals_CPU(ncomp=n_components)

        assert nips.fit(X)

        std = StandardScaler()
        X = std.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        message = "NIPS PCA not equal to pca from sklearn"
        decimalPlace = 2
        npt.assert_almost_equal(np.abs(X_pca), np.abs(
            nips.transform()), err_msg=message, decimal=1)

    def test_eig(self):
        n_components = 2
        X = np.random.randn(100, 30)
        # rng = np.random.RandomState(1)
        # X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
        t1 = time()

        nips = Nipals_CPU(ncomp=n_components)
        assert nips.fit(X)
        t2 = time()
        print('Total time for CPU NIPALS :', t2-t1)
        std = StandardScaler()
        X = std.fit_transform(X)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        npt.assert_allclose(pca.singular_values_, nips.eig, rtol=1)

  


if __name__ == '__main__':
    unittest.main()
