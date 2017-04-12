import partB
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import whiten



# Maximum order of mixands to consider
M_max = 2

# Maximum number of iterations 
N_iter = 8

# The training data is loaded
Xbar = sio.loadmat("../../data/training/XbarData.mat")['Xbar'] # N_R by 175
Ybar = sio.loadmat("../../data/training/YbarData.mat")['Ybar'][0][0][0][0][0].T # N_R by 8

# Whitening transformation
Xbar = whiten(Xbar)
Ybar = whiten(Ybar)

# Bic score container
bic = []

# for M in tqdm(range(2, M_max + 1)):
for M in tqdm(range(M_max, M_max + 1)):

    print "Model comprised of : " + str(M) + " mixands"
    omega, Nu, Sigma, Lambda, Mu, Psi,Gamma = partB.init(Xbar,Ybar,M)
   
    omega_old = np.copy(omega)

    for n in range(N_iter):
        print "Iteration " + str(n + 1) 

        # E-step : the responsibities are computed
        print "\tE-step"
        # Gamma = partB.E_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

    #     # M-step : the parameters are updated
    #     print "\tM-step"

    #     omega, Nu, Sigma, Lambda, Mu, Psi = partB.M_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)
        
    #     print " . Relative change: "
    #     print np.linalg.norm(omega - omega_old) / np.linalg.norm(omega_old)


    #    	omega_old = np.copy(omega)
    # # The BIC scored is computed and stored
    # bic += [[M,bic_score(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)]]