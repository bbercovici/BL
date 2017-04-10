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

def bic_score(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Evaluates the BIC score of the gaussian mixture model
	Inputs:
	-------
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- evaluated BIC score
	'''
	return len(omega) * np.log(len(Y)) - 2 * np.log(ICLL)

def gamma_Z_im(x_i,y_i,m,omega,Nu,Sigma,Lambda,Mu,Psi):
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
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	'''

	gamma = omega[m] * gaussian_pdf(y_i,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(x_i,Lambda[m,:,:].dot(y_i) + Mu[m,:],Psi[m,:,:])

	partial_sum = 0
	for m in range(len(omega)):
		partial_sum += omega_m * gaussian_pdf(y_i,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(x_i,Lambda[m,:,:].dot(y_i) + Mu[m,:],Psi[m,:,:])

	return gamma / partial_sum


def ICLL(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
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
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- evaluated ICLL
	'''

	icll = 0
	for i in range(len(Y)):

		icll_partial_sum = 0
		for m in range(len(omega)):

			icll_partial_sum += omega_m * gaussian_pdf(Y[i,:],Nu[m,:],Sigma[m,:,:]) * gaussian_pdf(X[i,:],Lambda[m,:,:].dot(Y[i,:]) + Mu[m,:],Psi[m,:,:])

		icll += np.log(icll_partial_sum)

	return icll


def nu_update(m,X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Computes the M-update for the m-th nu parameter
	Inputs:
	-------
	- m : mixand index
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated nu parameter 
	'''

	nu = np.zeros(8)
	resp_sum = 0
	for i in range(len(y_i)):
		nu += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * Y[i,:]
		resp_sum += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi)
	return nu / resp_sum



def Mu_update(m,X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Computes the M-update for the m-th Mu parameter
	Inputs:
	-------
	- m : mixand index
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated mu parameter 
	'''

	mu = np.zeros(125)

	resp_sum = 0
	for i in range(len(y_i)):
		mu += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * (X[i,:] - Lambda[m,:,:].dot(Y[i,:]))
		resp_sum += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi)
	return mu / resp_sum


def Sigma_update(m,X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Computes the m-update for the Sigma parameters
	Inputs:
	-------
	- m : mixand index
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Sigma parameter
	'''

	Sigma_m = np.zeros([8,8])
	resp_sum = 0
	for i in range(len(y_i)):
		Sigma_m += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * np.outer(Y[i,:] - Nu[m,:],Y[i,:] - Nu[m,:])
		resp_sum += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi)

	return Sigma_m / resp_sum

def Psi_update(m,X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Computes the m-update for the Psi parameters
	Inputs:
	-------
	- m : mixand index
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Psi parameter
	'''

	Psi_m = np.zeros([125,125])
	resp_sum = 0
	for i in range(len(y_i)):
		Psi_m += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * np.outer(X[i,:] - Lambda[m,:,:].dot(Y[i,:]) - Mu[m,:],X[i,:] - Lambda[m,:,:].dot(Y[i,:]) - Mu[m,:])
		resp_sum += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi)

	return Psi_m / resp_sum


def Lambda_update(m,X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
	'''
	Computes the m-update for the Lambda parameters
	Inputs:
	-------
	- m : mixand index
	- X : {x_i} (N_R x 125 )
	- Y : {y_i} (N_R x 8)
	- omega : {w_m} (M)
	- Nu : {Nu_m} (M x 8)
	- Sigma : {Sigma_m} (M x 8 x 8)
	- Lambda : {Lambda_m} (M x 125 x 8)
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated Lambda parameter
	'''

	LHS = np.zeros([125,8])
	RHS = np.zeros([8,8])

	for i in range(len(y_i)):
		LHS += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * np.outer( X[i,:] - Mu[m,:],Y[i,:])
		RHS += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi) * np.outer(Y[i,:],Y[i,:])

	return LHS.dot(np.linalg.inv(RHS))



def omega_update(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
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
	- Mu : {mu_m} (M x 125)
	- Psi : {Psi_m} (M x 125 x 125)
	Outputs:
	-------
	- updated omega parameter set
	'''
	resp_sum = 0

	for i in range(len(y_i)):

		resp_sum += gamma_Z_im(X[i,:],Y[i,:],m,omega,Nu,Sigma,Lambda,Mu,Psi)

	return resp_sum / len(Y)	



