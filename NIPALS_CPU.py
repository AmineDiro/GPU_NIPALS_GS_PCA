import numpy as np
from  scipy import linalg
import pandas as pd
import logging

# Implementét : https://cran.r-project.org/web/packages/nipals/vignettes/nipals_algorithm.html

# TODO : Write tests 

class Nipals():
   
    def __init__(self,ncomp=None, tol=0.0001, maxiter= 20):
        self.tol = tol 
        self.maxiter = maxiter
        self.ncomp=ncomp
    
    # TODO : Add class methods bach ndiro l'import nichane
    # Write class method bach normalizé lblane 

    @staticmethod 
    def normalize(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)        
        X = X - X_mean
        X = X / X_std
        return X

    def onestepcomp(self, X):
        # TODO : Check for missing values
        # Choose the column of X with highest var
        xvar = np.nanvar(X, axis=0, ddof=1)
        startcol_use = np.where(xvar == xvar.max())[0][0]
        th = X[:, startcol_use]

        it = 0
        while True:
            # loadings
            ph = X.T.dot(th) / np.sum(th*th)
            # Normalize
            ph = ph / np.sqrt(np.sum(ph*ph))

            # Scores
            th_old = th
            th = X.dot(ph) / sum(ph*ph)

            # Check convergence
            if np.sum((th-th_old)**2) < self.tol:
                break
            it += 1
            if it >= self.maxiter:
                raise RuntimeError("Convergence was not reached in {} iterations".format(self.maxiter) )
            
        return th, ph
    
    def fit(self,X,startcol=None):
        """
        fit method
        -------
        parametres 
        
        output
        ------
        """
        nr, nc = X.shape
        self.X = X.astype('float')
        self.X_PCA = self.X
        self.X_PCA = normalize(self.X_PCA)
        
        if self.ncomp is None:
            ncomp = min(self.X.shape)        
        else:
            try:          
                assert self.ncomp <= min(nr,nc) ,"can't will set this to{}".format(min(X.shape))
                ncomp = min(self.X.shape)        
            except AssertionError as msg:  
                print(msg)
                ncomp = min(self.X.shape)        
                
        # initialize outputs        
        eig = np.empty((ncomp,))
        loadings = np.empty((nc, ncomp))
        scores = np.empty((nr, ncomp))

        for comp in range(ncomp):
            # Calculate on full matrix
            th, ph = self.onestepcomp(self.X_PCA)    
            # Update X
            self.X_PCA = self.X_PCA - np.outer(th, ph)
            loadings[:, comp] = ph 
            scores[:, comp] = th
            eig[comp] = np.nansum(th*th)
            
        # Finalize eigenvalues and subtract from scores
        self.eig = pd.Series(np.sqrt(eig))
        
        # Convert results to DataFrames
        self.scores = scores #pd.DataFrame(scores, index=self.X.index, columns=["PC{}".format(i+1) for i in range(ncomp)])
        self.loadings = loadings # pd.DataFrame(loadings, index=self.X.columns, columns=["PC{}".format(i+1) for i in range(ncomp)])

        return 'DONE'
    
    