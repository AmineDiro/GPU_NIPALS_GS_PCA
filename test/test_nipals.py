from nipals.NIPALS_CPU import *
import sys
import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(
    0, '/Users/dihroussi/Google Drive/Documents/ENSAE/GPU/Projet GPU')


class TestNipals(unittest.TestCase):

    def test_pca(self):
        # generate data
        n_components = 2
        X = np.random.randn(100, 4)

        rng = np.random.RandomState(1)
        X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
        
        nips = Nipals(ncomp=n_components)

        assert nips.fit(X)

        std = StandardScaler()
        X = std.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        message = " Not equal to PCA"
        decimalPlace = 2
        # self.assertAlmostEqual(np.abs(X_pca).tolist(), np.abs(
        #     nips.transform().tolist()), decimalPlace,message)
        assert np.testing.assert_almost_equal(np.abs(X_pca),np.abs(nips.transform()),decimal =2)


    # def test_sklearn_pca(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
