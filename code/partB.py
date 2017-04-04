import numpy as np


def gaussian_pdf(arg,mean,cov):
	'''
	Evaluates a multivariate gaussian pdf 
	Inputs:
	------
	- arg : arguments at which the pdf is evaluated (N-by-1)
	- mean : mean of the pdf (N-by-1)
	- cov : covariance of the pdf (N-by-N)
	Outputs:
	-----
	- evaluated pdf
	'''

	return 1. / np.sqrt(np.linalg.det(2 * np.pi * cov)) * np.exp(-0.5 * np.inner(arg - mean,np.linalg.inv(cov).dot(arg - mean)))


def gamma_Z_im(x_i,y_i,m,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the responsibility of the m-th mixand for
	the i-th measurements
	Inputs:
	------
	- x_i : 125 x 1
	- y_i : 125 x 1
	- m : mixand index
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	'''

	gamma = omega[m] * gaussian_pdf(y_i,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(x_i,Lambda[m,:,:].dot(y_i) + mu[m,:],Psi[m,:,:])

	partial_sum = 0
	for m in range(len(omega)):
		partial_sum += omega_m * gaussian_pdf(y_i,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(x_i,Lambda[m,:,:].dot(y_i) + mu[m,:],Psi[m,:,:])

	return gamma / partial_sum


def ICLL(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''	
	Evaluates the ICLL
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- evaluated ICLL
	'''

	icll = 0
	for i in range(len(Y)):

		icll_partial_sum = 0
		for m in range(len(omega)):

			icll_partial_sum += omega_m * gaussian_pdf(Y[i,:],Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(X[i,:],Lambda[m,:,:].dot(Y[i,:]) + mu[m,:],Psi[m,:,:])

		icll += np.log(icll_partial_sum)

	return icll


def nu_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the nu parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated nu parameter set
	'''


def my_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the mu parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated mu parameter set
	'''

def Sigma_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the Sigma parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Sigma parameter set
	'''


def Psi_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the Psi parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Psi parameter set
	'''

def Lambda_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the Lambda parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Lambda parameter set
	'''

def omega_update(X,Y,omega,Nu,Sigma,Lambda,mu,Psi):
	'''
	Computes the m-update for the omega parameters
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated omega parameter set
	'''

