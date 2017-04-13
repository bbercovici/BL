import partB
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import whiten



# Maximum order of mixands to consider
M_min = 2
M_max = 17

# Maximum number of iterations 
N_iter_max = 50

# The training data is loaded
Xbar = sio.loadmat("../../data/training/XbarData.mat")['Xbar'] # N_R by 175
Ybar = sio.loadmat("../../data/training/YbarData.mat")['Ybar'][0][0][0][0][0].T # N_R by 8

# Whitening transformation
Xbar = whiten(Xbar)
Ybar = whiten(Ybar)

# Bic score container
bic = []

for M in tqdm(range(M_min, M_max + 1)):

    print "\nModel comprised of : " + str(M) + " mixands"
    omega, Nu, Sigma, Lambda, Mu, Psi,Gamma = partB.init(Xbar,Ybar,M)
    
    icll_old = partB.ICLL(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

    for n in range(N_iter_max):
        print "Iteration " + str(n + 1) 

        # E-step : the responsibities are computed
        print "\tE-step"
        Gamma = partB.E_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

        # M-step : the parameters are updated
        print "\tM-step"

        omega, Nu, Sigma, Lambda, Mu, Psi = partB.M_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)
        
        print "ICLL: "
        icll = partB.ICLL(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)
        print icll

        print "Relative change in ICLL (%): "
        change = np.abs(icll - icll_old) / np.abs(icll_old) * 100
        print str(change) + " %"

        if change < 0.01:
            break

        icll_old = icll
    
    # The BIC scored is computed and stored
    bic += [[M,partB.bic_score(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)]]