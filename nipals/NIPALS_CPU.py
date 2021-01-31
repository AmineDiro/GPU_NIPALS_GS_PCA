import numpy as np
from scipy import linalg
import pandas as pd
import logging

# Implementét : https://cran.r-project.org/web/packages/nipals/vignettes/nipals_algorithm.html

# TODO : Write tests


class Nipals_cpu():

    def __init__(self, ncomp=None, tol=0.00001, maxiter=500):
        self.tol = tol
        self.maxiter = maxiter
        self.ncomp = ncomp

    # TODO : Add class methods bach ndiro l'import nichane
    # Write class method bach normalizé lblane

    # TODO Fix this shitty job
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

        th = X[:,comp]

        it = 0
        while True:
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
            it += 1
            if it >= self.maxiter:
                raise RuntimeError(
                    "Convergence not reached in {} iterations".format(self.maxiter))
        eigh = np.sqrt(np.sum(th*th))
        return th, ph ,eigh

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
            th, ph, eigh = self.onestepcomp(self.X_PCA,comp)
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

    def transform(self):
        # Rertrieve the eigenvectors from score T= US where S is diag 
        U = self.scores/ self.eig.values
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
        self.fit(X,startcol=startcol)
        return self.transform()

