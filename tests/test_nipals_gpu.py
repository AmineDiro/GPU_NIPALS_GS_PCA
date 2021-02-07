from nipals.NIPALS_GPU import Nipals_GPU

import sys
import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy.testing as npt
from time import time


# sys.path.insert(
#    0, '/Users/dihroussi/Google Drive/Documents/ENSAE/GPU/Projet GPU')


class TestNipalsGPU(unittest.TestCase):

    def test_eig(self):
        X = np.random.randn(100, 30)
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

        npt.assert_allclose(pca.singular_values_, nips.eig, rtol=1)


if __name__ == '__main__':
    unittest.main()
