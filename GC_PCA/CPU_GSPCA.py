
import numpy as np
  

def SG_PCA(X,n,epsilon):
	R = np.copy(X)
	V = np.zeros((R.shape[0],R.shape[1]))
	Lambda = np.zeros((R.shape[0],1))
	U = np.zeros((X.shape[1],X.shape[1]))
	for k in range(n):
		mu = 0
		V[k,:] = R[k,:]
		while True:
			U[k,:] = np.dot(R.T,V)[k,:]
			if k>0:
				eigen = (U[k-1,:]*(U[k,:]).T)[0] / np.linalg.norm(U[k,:])
				A = np.dot( eigen , U[k,:]) 
				U[k,:] = U[k-1,:] - A
			L2 = np.linalg.norm(U[k,:])
			U[k,:] = U[k,:]/L2
			V[k,:] = np.dot(R,U)[k,:]
			if k>0:
				B = (V[k-1,:]*(V[k,:]).T)[0] / np.linalg.norm(V[k,:]) * V[k,:] 
				V[k,:] = V[k-1,:] - B
			Lambda[k] = np.linalg.norm(V[k,:])
			V[k,:] = V[k,:]/Lambda[k]
			if np.abs(Lambda[k]-mu) < epsilon:
				break
			mu = Lambda[k]
		R = R - np.dot(Lambda[k],np.dot(U[k,:],(V[k,:]).T))
	T = V * Lambda
	P = U
	return T,P,R