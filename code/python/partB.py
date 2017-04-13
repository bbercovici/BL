import numpy as np
from scipy.cluster.vq import vq, kmeans,kmeans2, whiten


def init(Xbar,Ybar,M):
    '''
    Initializes the parameters of the Gaussian sum
    formed with M mixands
    Inputs:
    ------
    - Xbar : {x_i} (N_R x 175)
    - Ybar : {y_i} (N_R x 8)
    - M : number of mixands
    Outputs:
    -------
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma
    '''
    # The GM parameters are initalized
    # Hard mixand assignment is performed through the k-means algorithm

    N_R = len(Ybar)

    Nu, mixand_index = kmeans2(Ybar,M,iter = 400)

    # Tally of owned points for each mixand
    count = np.zeros(M)
    for m in range(M):
        count[m] = sum([1 for i in range(N_R) if mixand_index[i] == m])

    ####################################################
    ######## The parameters are now initialized ########
    ####################################################


    

    ##################
    ###### Gamma #####
    ##################
    print "\t Initializing Gamma"
    Gamma = np.zeros([N_R,M])
    Gamma[range(len(mixand_index)),mixand_index] = 1

    ##################
    # Mixand weights #
    ##################
    print "\t Initializing weights"

    omega = np.zeros(M)

    for i in range(N_R):

        for m in range(0,M):

            if (mixand_index[i]) == m:
               omega[m] += 1

    omega = omega / N_R

   
    ##################
    ##### Sigma ######
    ##################
    print "\t Initializing Sigma"

    Sigma = np.zeros([M,8,8])
    m = 0
        
    for m in range(0,M):

        Sigma[m,:,:] = sum([np.outer(Ybar[i,:] - Nu[m,:], 
            Ybar[i,:] - Nu[m,:]) for i in range(N_R) if mixand_index[i] == m])
        Sigma[m,:,:] = Sigma[m,:,:] / count[m]
 

    ##################
    ####### Mu #######
    ##################
    print "\t Initializing Mu"

    Mu = np.zeros([M,175])

    # Suppose that X and Y are uncorrelated at first
    for m in range(0,M):
        Mu[m,:] = np.sum([ Xbar[i,:]  for i in range(N_R) if mixand_index[i] == m],axis = 0)
        Mu[m,:] = Mu[m,:] / count[m]

    ##################
    ##### Lambda #####
    ##################
    print "\t Initializing Lambda"
       
    Lambda = np.zeros([M,175,8])

    # for m in range(0,M):

    #     RHS = sum([np.outer(Ybar[i,:] , Ybar[i,:] ) for i in range(N_R) if mixand_index[i] == m])
    #     LHS = sum([np.outer(Xbar[i,:] - Mu[m,:], Ybar[i,:] ) for i in range(N_R) if mixand_index[i] == m])
    #     Lambda[m,:,:] = np.dot(LHS,np.linalg.inv(RHS))

    ##################
    ###### Psi #######
    ##################

    print "\t Initializing Psi"

    Psi = np.zeros([M,175,175])

    for m in range(0,M):

        # Suppose that X and Y are uncorrelated at first

        for i in range(0,N_R):
            if mixand_index[i] == m:
                Psi[m,:,:] += np.diag(np.diag(np.outer(Xbar[i,:] - Mu[m,:],
                    Xbar[i,:] - Mu[m,:])))

        Psi[m,:,:] = Psi[m,:,:] / count[m]

    return omega, Nu, Sigma, Lambda, Mu, Psi, Gamma







def E_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi):
    '''
    Computes the responsabilities of the Gaussian mixture for the observations X,Y
    Inputs:
    ------
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    Outputs:
    ------
    Gamma : N_R x M responsibilities
    '''
    N_R = len(Ybar)
    M = len(omega)

    Gamma = np.zeros([N_R,M])

    for m in range(M):
        Gamma[:,m] = omega[m] * gaussian_pdf_vec(Ybar,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf_vec(Xbar,(Lambda[m,:,:].dot(Ybar.T)).T + Mu[m,:],Psi[m,:,:])

     # Nan terms are found and reset



    Gamma = np.diag(1./np.sum(Gamma,axis = 1)).dot( Gamma )
  	
    nan_indices = np.array(range(len(Gamma)))[np.isnan(Gamma).sum(axis = 1) > 0]

    print nan_indices
    if len(nan_indices) > 0:
	    Gamma[nan_indices,0] = 1
	    Gamma[nan_indices,1:] = 0

    return Gamma


def M_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''    
    Updates the parameters of the Gaussian mixture
    Inputs:
    ------
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : Gamma : (N_R x M) responsibilities
    Outputs:
    -------
    - omega : {w_m} (M) (updated)
    - Nu : {Nu_m} (M x 8) (updated)
    - Sigma : {Sigma_m} (M x 8 x 8) (updated)
    - Lambda : {Lambda_m} (M x 175 x 8) (updated)
    - Mu : {mu_m} (M x 175) (updated)
    - Psi : {Psi_m} (M x 175 x 175)  (updated)
    '''

    sum_resp = np.sum(Gamma,axis = 0)
    M = len(omega)
    N_R = len(Ybar)

    ##################
    # Mixand weights #
    ##################
    print "\t Updating weights"

    omega = sum_resp / N_R

    ##################
    ####### Nu #######
    ##################
    print "\t Updating Nu"
    
    for m in range(M):
        Nu[m,:] = nu_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)

    ##################
    ##### Sigma ######
    ##################
    print "\t Updating Sigma"
    
    for m in range(0,M):
        Sigma[m,:,:] = Sigma_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)


    ##################
    ####### Mu #######
    ##################
    print "\t Updating Mu"

    for m in range(0,M):
        Mu[m,:] = Mu_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)

    ##################
    ##### Lambda #####
    ##################
    print "\t Updating Lambda"
    
    for m in range(0,M):
        Lambda[m,:,:] = Lambda_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)

    ##################
    ###### Psi #######
    ##################
    print "\t Updating Psi"

    for m in range(0,M):
        Psi[m,:,:] = Psi_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)

    return omega, Nu, Sigma, Lambda, Mu, Psi

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

    return 1. / np.sqrt(np.abs(np.linalg.det(2 * np.pi * cov))) * np.exp(- 0.5 * np.inner(arg - mean,np.linalg.inv(cov).dot(arg - mean)))

def gaussian_pdf_vec(arg,mean,cov):
    '''
    Evaluates a multivariate gaussian pdf 
    Inputs:
    ------
    - arg : arguments at which the pdf is evaluated (N - by - N_R)
    - mean : mean of the pdf (N - by - M)
    - cov : covariance of the pdf (M - by - M)
    Outputs:
    -----
    - evaluated pdf
    '''

    X = arg - mean
   
    return 1. / np.sqrt(np.linalg.det(2 * np.pi * cov)) * np.exp(-0.5 * np.diag(np.dot(X,np.linalg.inv(cov).dot(X.T))))


def bic_score(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
    '''
    Evaluates the BIC score of the gaussian mixture model
    Inputs:
    -------
    - X : {x_i} (N_R x 175 )
    - Y : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    Outputs:
    -------
    - evaluated BIC score
    '''

    return len(omega) * np.log(len(Y)) - 2 * ICLL(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi)




def ICLL(X,Y,omega,Nu,Sigma,Lambda,Mu,Psi):
    '''    
    Evaluates the ICLL
    Inputs:
    -------
    - X : {x_i} (N_R x 175 )
    - Y : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    Outputs:
    -------
    - evaluated ICLL
    '''

    icll_table = np.zeros([len(Y),len(omega)])
    for m in range(len(omega)):
        icll_table[:,m] = omega[m] * gaussian_pdf_vec(Y,Nu[m,:],Sigma[m,:,:]) * gaussian_pdf_vec(X,(Lambda[m,:,:].dot(Y.T)).T + Mu[m,:],Psi[m,:,:])

    icll = np.sum(np.log(np.sum(icll_table,axis = 1)))

    return icll


def nu_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''
    Computes the M-update for the m-th nu parameter
    Inputs:
    -------
    - m : mixand index
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : (N_R x M) 
    Outputs:
    -------
    - updated nu parameter 
    '''

    
    nu = Gamma[:,m] .dot(Ybar)
   
    return nu /  Gamma[:,m].sum()



   


def Mu_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''
    Computes the M-update for the m-th Mu parameter
    Inputs:
    -------
    - m : mixand index
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : (N_R x M) 

    Outputs:
    -------
    - updated mu parameter 
    '''

   
    mu = Gamma[:,m] .dot( (Xbar - (Lambda[m,:,:].dot(Ybar.T)).T))
   
    return mu / Gamma[:,m].sum()


def Sigma_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''
    Computes the m-update for the Sigma parameters
    Inputs:
    -------
    - m : mixand index
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : (N_R x M) 

    Outputs:
    -------
    - updated Sigma parameter
    '''

    Sigma_m = np.zeros([8,8])

    for i in range(len(Ybar)):
        Sigma_m += Gamma[i,m] * np.outer(Ybar[i,:] - Nu[m,:],Ybar[i,:] - Nu[m,:])

    return Sigma_m / Gamma[:,m].sum()

def Psi_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''
    Computes the m-update for the Psi parameters
    Inputs:
    -------
    - m : mixand index
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : (N_R x M) 

    Outputs:
    -------
    - updated Psi parameter
    '''

    Psi_m = np.zeros([175,175])


    for i in range(len(Ybar)):
        Psi_m += Gamma[i,m] * np.diag(np.diag(np.outer(Xbar[i,:] - Lambda[m,:,:].dot(Ybar[i,:]) - Mu[m,:],Xbar[i,:] - Lambda[m,:,:].dot(Ybar[i,:]) - Mu[m,:])))

    return Psi_m / Gamma[:,m].sum()


def Lambda_update(m,Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma):
    '''
    Computes the m-update for the Lambda parameters
    Inputs:
    -------
    - m : mixand index
    - Xbar : {x_i} (N_R x 175 )
    - Ybar : {y_i} (N_R x 8)
    - omega : {w_m} (M)
    - Nu : {Nu_m} (M x 8)
    - Sigma : {Sigma_m} (M x 8 x 8)
    - Lambda : {Lambda_m} (M x 175 x 8)
    - Mu : {mu_m} (M x 175)
    - Psi : {Psi_m} (M x 175 x 175)
    - Gamma : (N_R x M) 

    Outputs:
    -------
    - updated Lambda parameter
    '''

    LHS = np.zeros([175,8])
    RHS = np.zeros([8,8])

    for i in range(len(Ybar)):
        LHS += Gamma[i,m] * np.outer( Xbar[i,:] - Mu[m,:],Ybar[i,:])
        RHS += Gamma[i,m] * np.outer(Ybar[i,:],Ybar[i,:])

    return LHS.dot(np.linalg.inv(RHS))





