import partB
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import whiten



# Maximum order of mixands to consider
M_min = 2
M_max = 2

# Maximum number of iterations 
N_iter_max = 300

# Tolerance on ICLL
tol = 1e-3

# The training data is loaded
Xbar = sio.loadmat("../../../data/training/Xbar_R.mat")['Xbar_R'] # N_R by 175
Ybar = sio.loadmat("../../../data/training/Ybar_R.mat")['Ybar_R'][0][0][0][0][0].T # N_R by 8

# location_to_indices contains the look-up table associating each element in Xbar to its location/image of origin
# - len(location_to_indices) == number of locations, N_L
# - location_to_indices[l] == array of size N_S x N_i_l (N_S: number of samples. N_i_l: number of images for this location) 

# locations_to_indices_R = sio.loadmat("../../data/training/locations_to_indices.mat")['locations_to_indices'][0,0]

N_R = len(Ybar)


# Bic score container
bic = []

for M in tqdm(range(M_min, M_max + 1)):

    print "\nModel comprised of : " + str(M) + " mixands"
    omega, Nu, Sigma, Lambda, Mu, Psi, Gamma = partB.init(Xbar,Ybar,M)
    
    
    icll_old = partB.ICLL(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

    for n in range(N_iter_max):
        
        print "\n########## Iteration " + str(n + 1) +" ############"

        if n > 0:
            # E-step : the responsibities are computed
            print "\tE-step"
            Gamma = partB.Gamma(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

        # M-step : the parameters are updated
        print "\tM-step"

        omega, Nu, Sigma, Lambda, Mu, Psi = partB.M_step(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi,Gamma)
        
        icll = partB.ICLL(Xbar,Ybar,omega,Nu,Sigma,Lambda,Mu,Psi)

        print "\nICLL: " + str(icll)

        change = (icll - icll_old) / np.abs(icll_old) * 100
        print "Relative change in ICLL (%): " + str(change) + " %"

        if change < tol:
            break

        icll_old = icll
    
    # The BIC scored is computed and stored
    bic += [[M,partB.bic_score(N_R,icll,M)]]

    # The model is saved
    sio.savemat("model_M_" + str(M) + ".mat",{'omega':omega,
        'Nu':Nu,
        'Sigma':Sigma,
        'Lambda':Lambda,
        'Mu':Mu,
        'Psi':Psi})