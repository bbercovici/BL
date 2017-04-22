import os, inspect
import numpy as np
from numpy import linalg as la
from scipy.special import digamma, gammaln, gamma
from scipy.cluster.vq import kmeans2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ------------------------------------------ RESULTS PLOTTING ------------------------------------------#
def save_plot(str):
    """
    Defines where and how to save the plots with the results from part C
    """
    def define_plots_folder_path():
        """
        Defines the absolute path of the folder where plots are saved, according to its relative path to this file
        """
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        splitFolderName = 'code/'
        splitPath = path.split(splitFolderName)
        plotPath = splitPath[0] + 'report/FiguresC/'
        return plotPath

    plotPath = define_plots_folder_path()
    plt.rcParams['figure.figsize'] = 3.0, 3.0
    plt.rcParams.update({'font.size': 8})
    plt.savefig(plotPath + "/" + str + ".pdf", bbox_inches='tight')


def plot_lower_bound(L_vec, plt_color, plt_name, bool_plt_saved):
    """
    Plots the results from part C
    """
    plt.figure()
    plt.plot(L_vec, color= plt_color)
    plt.xlabel('Iteration, $it$')
    plt.ylabel('Lower bound value, ${L}$')
    plt.legend(['${L}$'])
    title = 'Lower bound evaluation for location: ' + plt_name
    if bool_plt_saved:
        save_plot(plt_name)
    else:
        plt.title(title)

# ------------------------------------------ MATLAB DATA LOADING ------------------------------------------#
def find_matlab_data_path():
    """
    Defines the absolute path of the matlab data according to its relative path to this file
    """
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    splitFolderName = 'Bayes/'
    splitPath = path.split(splitFolderName)
    matPath = splitPath[0]
    return matPath

def retrieve_location_data_points(matPath):
    """
    For each location L = {AVS, ORCCA, corridor_office, corridor_orcca} retrieves the N data points, where
    each data point Y_n is a [Dx1] array.
    :param: matPath: absolute path where Ybar.mat is stored
    :return: Y_AVS [NxD]
    :return: Y_ORCCA [NxD]
    :return: Y_corridor_office [NxD]
    :return: Y_corridor_orcca [NxD]
    """
    Ybar = loadmat(matPath + 'Ybar.mat')['Ybar']
    Y_AVS = Ybar['AVS'][0,0].T
    Y_ORCCA = Ybar['ORCCA'][0,0].T
    Y_corridor_office = Ybar['corridor_office'][0,0].T
    Y_corridor_orcca = Ybar['corridor_orcca'][0,0].T
    return Y_AVS, Y_ORCCA, Y_corridor_office, Y_corridor_orcca



# ------------------------------------------ LOWER BOUND COEFFICIENTS ------------------------------------------#

def compute_ln_C(alpha_vec):
    """
    Computes log  of the C coefficient in the Dirichlet distribution
    :param: alpha_vec [Mx1]
    :return: ln C
    """
    alpha_hat = np.sum(alpha_vec)
    ln_C = gammaln(alpha_hat)
    sum_ln_gamma = 0.0
    for k in range(0, len(alpha_vec)):
        sum_ln_gamma += gammaln(alpha_vec[k])
    ln_C -= sum_ln_gamma
    return ln_C


def compute_ln_B(W, v):
    """
    Computes log  of the B coefficient in the Wishart distribution
    :param: v [scalar]
    :param: W [DxD]
    :return: ln B
    """
    D = W.shape[0]
    (sign_W, logdet_W) = la.slogdet(W)
    ln_B = -0.5*v* sign_W*logdet_W - 0.5*v*D*np.log(2.0) - 0.25*D*(D-1.0)*np.log(np.pi)
    sum_ln_gamma = 0.0
    for i in range(0, D):
        arg_gamma = 0.5*(v + 1.0 - i)
        sum_ln_gamma += gammaln(arg_gamma)
    ln_B -= sum_ln_gamma
    return ln_B

def compute_wishart_entropy_H(W, v, E_ln_Sigma_k):
    """
    Computes entropy H of Wishart distribution
    :param: v [scalar]
    :param: W [DxD]
    :param: E_ln_Sigma_k = lnSigma_vec[k]
    :return: B
    """
    D = W.shape[0]
    ln_B = compute_ln_B(W, v)
    H = -ln_B - 0.5*(v - D - 1.0)*E_ln_Sigma_k + 0.5*v*D
    return H


# ------------------------------------------ LOWER BOUND EXPECTATIONS ------------------------------------------#
def compute_L_q_MuSigma(B_vec, m_vec, v_vec, W_vec, lnSigma_vec):
    """
    Computes E[ln q*(Mu, Sigma)]
    :param: B_vec [Mx1]
    :param: m_vec [MxD]
    :param: v_vec [Mx1]
    :param: W_vec Mx[D-by-D]
    :param: lnSigma_vec [Mx1] stack of E[ln(|Sigma_k|)] values for each GM k

    :return: L_q_MuSigma = E[ln q*(Mu, Sigma)]
    """
    L_q_MuSigma = 0.0
    D = m_vec.shape[1]
    for k in range(0, len(lnSigma_vec)):
        W_k = W_vec[k * D:(k + 1) * D, :]
        H_k = compute_wishart_entropy_H(W_k, v_vec[k], lnSigma_vec[k])
        L_q_MuSigma += 0.5*lnSigma_vec[k] + 0.5*D*np.log(B_vec[k]/(2.0*np.pi)) - 0.5*D - H_k
    return L_q_MuSigma

def compute_L_q_Pi(alpha_vec, lnPi_vec):
    """
    Computes E[ln(q*(Pi)]
    :param: alpha_vec [1xM]  array w/ updated alpha_k hyper-parameters for each gaussian mixture k
    :param: lnPi_vec [Mx1] stack of E[ln(pi_k)] values for each GM k
    :return: L_q_Pi = E[ln(q*(Pi)]
    """
    L_q_Pi = compute_ln_C(alpha_vec)
    for k in range(0, len(alpha_vec)):
        L_q_Pi += (alpha_vec[k] - 1.0)*lnPi_vec[k]
    return L_q_Pi

def compute_L_q_Z(R_vec):
    """
    Computes E[ln(q*(Z)]
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n
    :return: L_q_Z = E[ln(q*(Z)]
    """
    L_q_Z = 0.0
    for n in range(0, R_vec.shape[0]):
        for k in range(0, R_vec.shape[1]):
            L_q_Z += R_vec[n, k] * np.log(R_vec[n, k])
    return L_q_Z

def compute_L_p_MuSigma(B0_prior, m0_vec_prior, v0_prior, W0_prior, B_vec, m_vec, v_vec, W_vec, lnSigma_vec):
    """
    Computes E[ln P(Mu, Sigma)]

    :param: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :param: m0_vec_prior [1xD]: initial mean vector of apriori Normal distrib
    :param: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :param: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib

    :param: B_vec [Mx1]
    :param: m_vec [MxD]
    :param: v_vec [Mx1]
    :param: W_vec Mx[D-by-D]
    :param: lnSigma_vec [Mx1] stack of E[ln(|Sigma_k|)] values for each GM k

    :return: L_p_MuSigma = E[ln P(Mu, Sigma)]
    """
    M = len(lnSigma_vec)
    D = m_vec.shape[1]
    ln_B0 = compute_ln_B(W0_prior, v0_prior)
    L_p_MuSigma = M*ln_B0 + 0.5*(v0_prior - D - 1.0)*np.sum(lnSigma_vec)
    W0_inv = la.inv(W0_prior)
    sum_k1 = 0.0
    sum_k2 = 0.0
    for k in range(0, M):
        W_k = W_vec[k * D:(k + 1) * D, :]
        sum_k1 += v_vec[k] * np.trace(np.dot(W0_inv, W_k))

        diff_k = m_vec[k, :] - m0_vec_prior
        sum_k2 += (D*np.log(B0_prior / (2.0*np.pi)) + lnSigma_vec[k] - D*B0_prior/B_vec[k] -
                   B0_prior*v_vec[k]*np.dot(diff_k, np.dot(W_k, diff_k)))

    L_p_MuSigma += (-0.5*sum_k1 + 0.5*sum_k2)
    return L_p_MuSigma

def compute_L_p_Pi(alpha_vec_prior, lnPi_vec):
    """
    Computes E[ln(P(Pi)]
    :param: alpha_vec_prior [Mx1]  array w/ initial alpha_k hyper-parameters for each gaussian mixture k
    :param: lnPi_vec [Mx1] stack of E[ln(pi_k)] values for each GM k

    :return: L_p_Pi = E[ln(P(Pi)]
    """
    ln_C0 = compute_ln_C(alpha_vec_prior)
    alpha_0 = alpha_vec_prior[0] #alpha_0 = np.sum(alpha_vec_prior)
    L_p_Pi = ln_C0 + (alpha_0 - 1.0) * np.sum(lnPi_vec)
    return L_p_Pi


def compute_L_p_Z(R_vec, lnPi_vec):
    """
    Computes E[ln(P(Z|...)]
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n
    :param: lnPi_vec [Mx1] stack of E[ln(pi_k)] values for each GM k

    :return: L_p_Z = E[ln(P(Z|...)]
    """
    L_p_Z = 0.0
    for n in range(0, R_vec.shape[0]):
        for k in range(0, R_vec.shape[1]):
            L_p_Z +=  R_vec[n, k] * lnPi_vec[k]
    return L_p_Z

def compute_L_p_Y(B_vec, m_vec, v_vec, W_vec, YHat_vec, S_vec, R_vec, lnSigma_vec):
    """
    Computes E[ln(P(Y|...)]
    :param: B_vec [Mx1]
    :param: m_vec [MxD]
    :param: v_vec [Mx1]
    :param: W_vec Mx[D-by-D]

    :param: YHat_vec [MxD] array of the [1xD] y_mean for each gaussian mixture k
    :param: S_vec Mx[D-by-D] array of the [DxD] y_covariance for each gaussian mixture k
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n

    :param: lnPi_vec [Mx1] stack of E[ln(pi_k)] values for each GM k
    :param: lnSigma_vec [Mx1] stack of E[ln(|Sigma_k|)] values for each GM k

    :return: L_p_Y = E[ln(P(Y|...)]
    """
    L_p_Y = 0.0
    D = YHat_vec.shape[1] # Dimensionality
    for k in range(0, R_vec.shape[1]):
        N_k = np.sum(R_vec[:, k])
        W_k = W_vec[k*D:(k+1)*D, :]
        S_k = S_vec[k*D:(k+1)*D, :]
        diff_k = YHat_vec[k, :] - m_vec[k, :]

        bracket_k = lnSigma_vec[k] - D/B_vec[k] - v_vec[k]*np.trace(np.dot(S_k, W_k)) - \
                    v_vec[k]* np.dot(diff_k, np.dot(W_k, diff_k)) - D*np.log(2.0*np.pi)
        L_p_Y += N_k * bracket_k
    L_p_Y *= 0.5
    return L_p_Y


# ------------------------------------------ LOWER BOUND  ------------------------------------------#
def compute_lower_bound_L(alpha_vec_prior, B0_prior, m0_vec_prior, v0_prior, W0_prior,
                          alpha_vec, B_vec, m_vec, v_vec, W_vec, YHat_vec, S_vec, R_vec, lnPi_vec, lnSigma_vec):
    """
    Computes the variational lower bound L
    :param: alpha_vec_prior [Mx1]  array w/ initial alpha_k values of apriori Dirichlet distrib for each GM k
    :param: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :param: m0_vec_prior [1xD]: initial mean vector of apriori Normal distrib
    :param: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :param: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib

    :param: alpha_vec [Mx1] array w/ updated alpha_k hyper-parameters of Dirichlet distrib for each GM k
    :param: B_vec [Mx1]  array w/ updated Beta hyper-parameters of Normal distrib for each GM k
    :param: m_vec [MxD] array w/ updated mean vectors of Normal distrib for each GM k
    :param: v_vec [Mx1] array w/ updated DOF values of Wishart distrib for each GM k
    :param: W_vec Mx[D-by-D] array w/ updated covar matrices of Wishart distrib

    :param: YHat_vec [MxD] array of the [1xD] y_mean for each gaussian mixture k
    :param: S_vec Mx[D-by-D] array of the [DxD] y_covariance for each gaussian mixture k
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n

    :param: lnPi_vec [Mx1] stack of E[ln(pi_k)] values for each GM k
    :param: lnSigma_vec [Mx1] stack of E[ln(|Sigma_k|)] values for each GM k
    """
    L_p_Y = compute_L_p_Y(B_vec, m_vec, v_vec, W_vec, YHat_vec, S_vec, R_vec, lnSigma_vec)
    L_p_Z = compute_L_p_Z(R_vec, lnPi_vec)
    L_p_Pi = compute_L_p_Pi(alpha_vec_prior, lnPi_vec)
    L_p_MuSigma = compute_L_p_MuSigma(B0_prior, m0_vec_prior, v0_prior, W0_prior, B_vec, m_vec, v_vec, W_vec, lnSigma_vec)
    L_q_Z = compute_L_q_Z(R_vec)
    L_q_Pi = compute_L_q_Pi(alpha_vec, lnPi_vec)
    L_q_MuSigma = compute_L_q_MuSigma(B_vec, m_vec, v_vec, W_vec, lnSigma_vec)

    L = L_p_Y + L_p_Z + L_p_Pi + L_p_MuSigma - L_q_Z - L_q_Pi - L_q_MuSigma

    def print_results():
        print 'L_p_Y = ', L_p_Y
        print 'L_p_Z = ', L_p_Z
        print 'L_p_Pi = ', L_p_Pi
        print 'L_p_MuSigma = ', L_p_MuSigma
        print 'L_q_Z = ', L_q_Z
        print 'L_q_Pi = ', L_q_Pi
        print 'L_q_MuSigma = ', L_q_MuSigma
    #print_results()
    return L


# ------------------------------------------ RESPONSIBILITIES EVALUATION  ------------------------------------------#
# All the following methods are used to compute optimal solution q*(Z), which depends on moments evaluated
# with respect to the distribution of the other variables (parameters).
# This means that the current distributions over model params are used to evaluate responsibilities.

def compute_probabilities_P(alpha_vec, B_vec, m_vec, v_vec, W_vec, Y_vec):
    """
    Computes the expectations: E[ln(pi_k)], E[ln(|Sigma_k|)], E(y_n, mu_k)[...]
    :param: alpha_vec [Mx1]  array of alpha parameters for the Dirichlet distrib of each GM k
    :param: B_vec [Mx1] array of current Beta params for the Normal distrib of each GM k
    :param: m_vec [MxD] array of current [Dx1] m params for the Normal distrib of each GM k
    :param: v_vec [Mx1] array of current v params for Wishart distrib of each GM k
    :param: W_vec Mx[D-by-D] stack of current [DxD] covar matrices for each GM k
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n

    :return: P_vec [NxM] double-array w/ all probabilities p_nk for each gaussian mixture k of each data point n
    :return: lnPi_vec [Mx1]stack of E[ln(Pi_k)] values for each GM k
    :return: lnSigma_vec [Mx1] stack of E[ln(|Sigma_k|)] values for each GM k
    """
    def compute_E_ln_Sigma(v_k, W_k):
        """
        Computes the individual expectation  E[ln(|Sigma_k|)]
        :param: v_k [scalar] DOF of Wishart distrib for GM k
        :param: W_k [D-by-D] covar matrix of Wishart distrib for GM k
        :return: E[ln(|Sigma_k|)]
        """
        D = W_k.shape[0]
        ln_W_k = np.log(la.det(W_k))
        Dln2 = D*np.log(2)
        E_ln_Sigma_k = 0.0
        for i in range(0, D):
            E_ln_Sigma_k += digamma(0.5*(v_k + 1.0 - i)) + Dln2 + ln_W_k
        return E_ln_Sigma_k

    lnPi_vec = np.zeros(len(alpha_vec)) # [Mx1]
    lnSigma_vec = np.zeros(len(alpha_vec)) # [Mx1]
    P_vec = np.zeros([Y_vec.shape[0], len(alpha_vec)]) # [NxM]
    D = Y_vec.shape[1] # dimensionality
    alpha_hat = np.sum(alpha_vec)
    digamma_hat = digamma(alpha_hat)
    # Compute nk-independent term of P_nk
    P = 0.5*D*np.log(2)
    for k in range(0, len(alpha_vec)):
        W_k = W_vec[k*D:(k+1)*D, :]
        E_lnPi_k = digamma(alpha_vec[k])  - digamma_hat
        E_lnSigma_k = compute_E_ln_Sigma(v_vec[k], W_k)
        # Store variables
        lnPi_vec[k] = E_lnPi_k
        lnSigma_vec[k] = E_lnSigma_k
        # Compute k-dependent terms of P_nk
        P_k = E_lnPi_k + 0.5*E_lnSigma_k - P
        for n in range(0, Y_vec.shape[0]):
            diff_nk = Y_vec[n, :] - m_vec[k, :]
            E_nk = D / B_vec[k] + v_vec[k] * np.dot(diff_nk, np.dot(W_k, diff_nk))
            P_nk = P_k - 0.5*E_nk
            P_vec[n, k] = P_nk
    return P_vec, lnPi_vec, lnSigma_vec


def compute_responsibilities_R(P_vec):
    """
    Computes the responsibilities r_nk
    :param: P_vec [NxM] double-array w/ all probabilities p_nk for each gaussian mixture k of each data point n
    :return: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n
    """
    R_vec = np.full_like(P_vec, 0.0)
    for n in range(0, P_vec.shape[0]):
        sum_p_nj = np.sum(P_vec[n, :])
        for k in range(0, P_vec.shape[1]):
            r_nk = P_vec[n, k] * 1.0 / sum_p_nj
            R_vec[n, k] = r_nk
    return R_vec


def compute_obs_data_statistics(R_vec, Y_vec):
    """
    Computes the statistics of the observed data set evaluated wrt to the responsibilities
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n
    :return: YHat_vec [MxD] array of the [1xD] y_mean for each gaussian mixture k
    :return: S_vec Mx[D-by-D] array of the [DxD] y_covariance for each gaussian mixture k
    """
    N_k_thresh = 1E-3
    YHat_vec = np.zeros([R_vec.shape[1], Y_vec.shape[1]]) # [MxD]
    S_vec = np.zeros([R_vec.shape[1]*Y_vec.shape[1], Y_vec.shape[1]]) # [M-by-D xD]
    #S = np.array([])
    for k in range(0, R_vec.shape[1]):
        N_k = np.sum(R_vec[:, k])

        y_k = np.full(Y_vec.shape[1], 0.0) # [Dx1]
        S_k = np.full([Y_vec.shape[1], Y_vec.shape[1]], 0.0) # [DxD]
        for n in range(0, R_vec.shape[0]):
            y_k += R_vec[n, k] * Y_vec[n, :]
            S_k += R_vec[n, k] * np.outer(Y_vec[n, :] - y_k, Y_vec[n, :] - y_k)

        if N_k < N_k_thresh:
            print 'N_k limited!'
            y_k *= 1.0 / N_k_thresh
            S_k *= 1.0 / N_k_thresh
        else:
            y_k *= 1.0 / N_k
            S_k *= 1.0 / N_k

        YHat_vec[k, :] = y_k
        S_vec[k*Y_vec.shape[1]:(k+1)*Y_vec.shape[1], :] = S_k
        #S = np.append(S, S_k)
    return YHat_vec, S_vec


# ------------------------------------------ DISTRIBUTIONS OVER MODEL PARAMS ------------------------------------------#
# All the following methods are used to compute optimal solutions q*(Pi) and q*(Mu, Sigma).
# This means that the current responsibilities are considered fixed and
# used to recompute the variational distributions over the model parameters.

def compute_dirichlet_hyperparams(alpha_vec_prior, R_vec):
    """
    Computes the new alpha parameters of the variational Dirichlet distribution
    :param: alpha_vec_prior [1xM]  array w/ initial alpha_k hyper-parameters for each gaussian mixture k
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n
    :return: alpha_vec [1xM]  array w/ updated alpha_k hyper-parameters for each gaussian mixture k
    """
    alpha_vec = np.zeros_like(alpha_vec_prior)
    for k in range(0, R_vec.shape[1]):
        N_k = np.sum(R_vec[:, k])
        alpha_k = alpha_vec_prior[k] + N_k
        alpha_vec[k] = alpha_k
    return alpha_vec

def compute_normal_wishart_hyperparams(B0_prior, m0_vec_prior, v0_prior, W0_prior, YHat_vec, S_vec, R_vec):
    """
    Computes the new m and Beta params of the Normal distribution, and
    Computes the new v and W params of the Wishart distribution.
    :param: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :param: m0_vec_prior [1xD]: initial mean vector of apriori Normal distrib
    :param: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :param: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib
    :param: YHat_vec [MxD] array of the [1xD] y_mean for each gaussian mixture k
    :param: S_vec Mx[D-by-D] array of the [DxD] y_covariance for each gaussian mixture k
    :param: R_vec [NxM] double-array w/ all responsibilities r_nk for each gaussian mixture k of each data point n

    :return: B_vec [Mx1] array of updated Beta params for the Normal distrib of each GM k
    :return: m_vec [MxD] array of updated [Dx1] m params for the Normal distrib of each GM k
    :return: v_vec [Mx1] array of updated v params for Wishart distrib of each GM k
    :return: W_vec Mx[D-by-D] stack of updated [DxD] covar matrices for each GM k
    """
    B_vec = np.zeros(R_vec.shape[1]) # [Mx1]
    m_vec = np.zeros([R_vec.shape[1], YHat_vec.shape[1]]) # [MxD]
    v_vec = np.zeros(R_vec.shape[1]) # [Mx1]
    W_vec = np.zeros([R_vec.shape[1]*YHat_vec.shape[1], YHat_vec.shape[1]]) # Mx[D-by-D]

    W0_inv = la.inv(W0_prior)
    for k in range(0, R_vec.shape[1]):
        N_k = np.sum(R_vec[:, k])
        B_k = B0_prior + N_k
        m_k = (1.0/B_k) * (B0_prior*m0_vec_prior + N_k*YHat_vec[k, :])
        W_k_inv = W0_inv + N_k*S_vec[k*YHat_vec.shape[1]:(k+1)*YHat_vec.shape[1], :] + \
                 B0_prior*N_k/(B0_prior + N_k)*np.outer(YHat_vec[k, :] - m0_vec_prior, YHat_vec[k, :] - m0_vec_prior)
        v_k = v0_prior + N_k
        # Store vars
        B_vec[k] = B_k
        m_vec[k, :] = m_k
        v_vec[k] = v_k
        W_vec[k*YHat_vec.shape[1]:(k+1)*YHat_vec.shape[1], :] = la.inv(W_k_inv)
    return B_vec, m_vec, v_vec, W_vec


# ------------------------------------------ INITIALIZATION ------------------------------------------#

def generate_initial_hyperparam_values(M, D):
    """
    Initializes the hyper-parameter values for each location
    :param: M: number of Gaussian Mixture models
    :param: D: dimensionality of the observed data set

    :return: alpha_vec_prior [Mx1]  array w/ initial alpha_k values of apriori Dirichlet distrib for each GM k
    :return: m0_vec_prior [1xD]: initial mean vector of apriori Normal distrib
    :return: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :return: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :return: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib
    :return: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n
    :return: W_vec Mx[D-by-D] stack of initial [DxD] covar matrices for each GM k
    """
    alpha_vec_prior = np.ones(M) # [Mx1] array of alpha_k parameters for each Dirichlet distribution
    # alpha_vec_prior *= (1.0/ np.sum(alpha_vec_prior)) # normalized
    m0_vec_prior = np.zeros(D) # [Dx1] mean
    B0_prior = 1.0 # [scalar]
    v0_prior = D # [scalar]
    W0_prior = np.identity(D) # [DxD]

    def print_output():
        print 'alpha_vec_prior = ', alpha_vec_prior
        print 'm0_vec_prior = ', m0_vec_prior
        print 'B0_prior = ', B0_prior
        print 'v0_prior = ', v0_prior
        print 'W0_prior = ', W0_prior

    return alpha_vec_prior, m0_vec_prior, B0_prior, v0_prior, W0_prior


def generate_initial_hyperparam_vectors(alpha_vec_prior, B0_prior, v0_prior, W0_prior, Y_vec):

    """
    Initializes the hyper-parameter vectors required to start the Variational Bayes iteration

    :param: alpha_vec_prior [Mx1]  array w/ initial alpha_k values of apriori Dirichlet distrib for each GM k
    :param: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :param: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :param: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n

    :return: alpha_vec [Mx1] array of initial alpha hyper-parameters for the Dirichlet distrib of each GM  k
    :return: B_vec [Mx1] array of initial Beta params for the Normal distrib of each GM k
    :return: m_vec [MxD] array of initial [Dx1] m params for the Normal distrib of each GM k
    :return: v_vec [Mx1] array of initial v params for Wishart distrib of each GM k
    :return: W_vec Mx[D-by-D] stack of initial [DxD] covar matrices for each GM k
    """
    M = len(alpha_vec_prior) # number of Gaussian Mixtures
    D = Y_vec.shape[1] # dimensionality of observed data
    alpha_vec = alpha_vec_prior.copy() # [Mx1]
    B_vec = np.full_like(alpha_vec_prior, B0_prior) # [Mx1]
    (m_vec, labels) = kmeans2(Y_vec, M) # classify the given data set Y into M clusters and retrieve their means
    v_vec =  np.full_like(alpha_vec_prior, v0_prior) # [Mx1]
    W_vec = np.zeros([M*D, D]) # Mx[D-by-D]
    for k in range(0, M):
        W_vec[k*D:(k+1)*D, :] = W0_prior

    def print_output():
        print 'alpha_vec = ', alpha_vec
        print 'B_vec = ', B_vec
        print 'm_vec = ', m_vec
        print 'm_vec.shape = ', m_vec.shape
        print 'v_vec = ', v_vec
        print 'W_vec = ', W_vec
        print 'W_vec.shape = ', W_vec.shape

    return alpha_vec, B_vec, m_vec, v_vec, W_vec


# ------------------------------------------ VARIATIONAL OPTIMIZATION ------------------------------------------#

def optimize_variational_distributions(alpha_vec_prior, B0_prior, m0_vec_prior, v0_prior, W0_prior, Y_vec):

    """
    Variational Bayes iterative process
    :param: alpha_vec_prior [Mx1]  array w/ initial alpha_k values of apriori Dirichlet distrib for each GM k
    :param: B0_prior [scalar]: initial Beta hyper-parameters of apriori Normal distrib
    :param: m0_vec_prior [1xD]: initial mean vector of apriori Normal distrib
    :param: v0_prior [scalar]: initial DOF value of apriori Wishart distrib
    :param: W0_prior [DxD]: initial covar matrix of apriori Wishart distrib
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n
    :return: L_vec: array with lower bound values of the VB for data set Y_vec until convergence or max it achieved
    """

    it_max = 1E4 # 1E8 # maxim number of iterations before stopping the sim
    tol_dL = 1E-3 # tolerance for lower bound L convergence

    alpha_vec, B_vec, m_vec, v_vec, W_vec = \
        generate_initial_hyperparam_vectors(alpha_vec_prior, B0_prior, v0_prior, W0_prior, Y_vec) # m0_vec_prior not needed

    L_vec = []
    for i in range(0, int(it_max)):
        # Step 1: evaluate responsibilities with current distributions over model params.
        P_vec, lnPi_vec, lnSigma_vec = compute_probabilities_P(alpha_vec, B_vec, m_vec, v_vec, W_vec, Y_vec)
        R_vec = compute_responsibilities_R(P_vec)
        YHat_vec, S_vec = compute_obs_data_statistics(R_vec, Y_vec)

        # Step 2: recompute the variational distributions over the model params with current responsibilities
        alpha_vec = compute_dirichlet_hyperparams(alpha_vec_prior, R_vec)
        B_vec, m_vec, v_vec, W_vec = \
            compute_normal_wishart_hyperparams(B0_prior, m0_vec_prior, v0_prior, W0_prior, YHat_vec, S_vec, R_vec)

        # Check convergence: compute lower bound L
        L = compute_lower_bound_L(alpha_vec_prior, B0_prior, m0_vec_prior, v0_prior, W0_prior,
                              alpha_vec, B_vec, m_vec, v_vec, W_vec, YHat_vec, S_vec, R_vec, lnPi_vec, lnSigma_vec)
        L_vec.append(L)
        print 'L = ', L
        if i > 0:
            dL = L_vec[i] - L_vec[i-1]
            if dL < tol_dL:
                print 'Lower bound L converged!'
                for k in range(0, R_vec.shape[1]):
                    N_k = np.sum(R_vec[:, k])
                    print 'k = ', k, '\t N_k = ', N_k
                break
    return L_vec


# ------------------------------------------ INTERFACES ------------------------------------------#

def main_VB(M, Y_vec):
    """
    Interface for the Variational Bayes iterative process.
    This interface defines initial parameters that in principle would be tuned according to the given data set Y_vec.
    For the sake of simplicity, optimize_variational_distributions(...) is right now being called with the same input.
    :param: M: maximum number of Gaussian Mixtures for data set Y_vec
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n
    :return: L_vec: array with lower bound values of the VB for data set Y_vec until convergence or max it achieved
     """
    alpha_vec_prior, m0_vec_prior, B0_prior, v0_prior, W0_prior = generate_initial_hyperparam_values(M, Y_vec.shape[1])
    L_vec = optimize_variational_distributions(alpha_vec_prior, B0_prior, m0_vec_prior, v0_prior, W0_prior, Y_vec)
    return L_vec

def run_location_ORCCA(Y_ORCCA, bool_plt_saved):
    """
    Handles the function main_VB and the plotting of results for location ORCCA.
    :param: Y_ORCCA: observed data set for location ORCCA
    :param: bool_plt_saved: boolean var specifying if plots should be saved
    """
    print '\n' + 'Running location: ORCCA'
    M = 12  # number of Gaussian Mixture models
    L_vec = main_VB(M, Y_ORCCA)
    plot_lower_bound(L_vec, 'lightgreen', 'ORCCA', bool_plt_saved)

def run_location_AVS(Y_AVS, bool_plt_saved):
    """
    Handles the function main_VB and the plotting of results for location AVS.
    :param: Y_AVS: observed data set for location AVS
    :param: bool_plt_saved: boolean var specifying if plots should be saved
    """
    print '\n' + 'Running location: AVS'
    M = 12  # number of Gaussian Mixture models
    L_vec = main_VB(M, Y_AVS)
    plot_lower_bound(L_vec, 'dodgerblue', 'AVS', bool_plt_saved)

def run_location_corridor_office(Y_corridor_office, bool_plt_saved):
    """
    Handles the function main_VB and the plotting of results for location corridor office.
    :param: Y_corridor_office: observed data set for location corridor office
    :param: bool_plt_saved: boolean var specifying if plots should be saved
    """
    print '\n' + 'Running location: corridor office'
    M = 12  # number of Gaussian Mixture models
    L_vec = main_VB(M, Y_corridor_office)
    plot_lower_bound(L_vec, 'red', 'corridor_office', bool_plt_saved)

def run_location_corridor_ORCCA(Y_corridor_orcca, bool_plt_saved):
    """
    Handles the function main_VB and the plotting of results for location corridor ORCCA.
    :param: Y_corridor_orcca: observed data set for location corridor ORCCA
    :param: bool_plt_saved: boolean var specifying if plots should be saved
    """
    print '\n' + 'Running location: corridor ORCCA'
    M = 12  # number of Gaussian Mixture models
    L_vec = main_VB(M, Y_corridor_orcca)
    plot_lower_bound(L_vec, 'magenta', 'corridor_ORCCA', bool_plt_saved)


# ------------------------------------------ MAIN ------------------------------------------#

if __name__ == "__main__":
    bool_plt_saved = False
    matPath = find_matlab_data_path()
    Y_AVS, Y_ORCCA, Y_corridor_office, Y_corridor_orcca = retrieve_location_data_points(matPath)
    run_location_ORCCA(Y_ORCCA, bool_plt_saved)
    run_location_AVS(Y_AVS, bool_plt_saved)
    run_location_corridor_office(Y_corridor_office, bool_plt_saved)
    run_location_corridor_ORCCA(Y_corridor_orcca, bool_plt_saved)
    plt.show()


















