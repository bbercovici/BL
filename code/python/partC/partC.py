import numpy as np
from numpy import linalg as la
from scipy.special import digamma


N = 4 # number of training data points
M = 2 # number of Gaussian Mixture models
D = 8
# Observed data set
Y = np.zeros([N, D])
# Hidden variables
Z = np.zeros([N, M])

# Hyper-parameters
alpha_vec_prior = np.ones(M) # [Mx1]
alpha_0 = np.sum(alpha_vec_prior)
m0_vec_prior = np.zeros(Y.shape[1]) # [Dx1]
B0_prior = 1.0 # scalar
v0_prior = Y.shape[1] # scalar
W0_prior = np.identity(Y.shape[1]) # [DxD]


# Pi parameters
pi_vec = np.ones(M)
pi_sum = np.sum(pi_vec)
pi_vec *= 1.0 / pi_sum
# mu parameter
mu = np.zeros(Y.shape[1])
# Sigma parameter
Sigma = np.identity(Y.shape[1])


def compute_probabilities_P(alpha_vec, B_vec, m_vec, v_vec, W_vec, Y_vec):
    """
    Computes the expectations: E[ln(pi_k)], E[ln(|Sigma_k|)], E(y_n, mu_k)[...]
    :param: alpha_vec [Mx1]  array of alpha parameters for the Dirichlet distrib of each GM k
    :param: B_vec [Mx1] array of current Beta params for the Normal distrib of each GM k
    :param: m_vec [MxD] array of current [Dx1] m params for the Normal distrib of each GM k
    :param: v_vec [Mx1] array of current v params for Whishart distrib of each GM k
    :param: W_vec Mx[D-by-D] stack of current [DxD] covar matrices for each GM k
    :param: Y_vec [NxD]  array of the [1xD] observation variables y_n for each data point n

    :return: P_vec [NxM]
    """
    def compute_E_ln_Sigma(v_k, W_k):
        """
        Computes the individual expectation  E[ln(|Sigma_k|)]
        :param: v_k [scalar] DOF of Whishart distrib for GM k
        :param: W_k [D-by-D] covar matrix of Whishart distrib for GM k
        :return: E[ln(|Sigma_k|)]
        """
        D = W_k.shape[0]
        ln_W_k = np.log(la.det(W_k))
        Dln2 = D*np.log(2)
        E_ln_Sigma_k = 0.0
        for i in range(0, D):
            E_ln_Sigma_k += digamma(0.5*(v_k + 1 - i)) + Dln2 + ln_W_k
        return E_ln_Sigma_k

    P_vec = np.zeros([Y_vec.shape[0], len(alpha_vec)]) # [NxM]
    D = Y_vec.shape[1] # dimensionality
    alpha_hat = np.sum(alpha_vec)
    digamma_hat = digamma(alpha_hat)
    P = 0.5*D*np.log(2)
    for k in range(0, len(alpha_vec)):
        W_k = W_vec[k*D:(k+1)*D, :]
        E_lnPi_k = digamma(alpha_vec[k])  - digamma_hat
        E_lnSigma_k = compute_E_ln_Sigma(v_vec[k], W_k)
        P_k = E_lnPi_k + 0.5*E_lnSigma_k - P
        for n in range(0, Y_vec.shape[0]):
            diff_nk = Y_vec[n, :] - m_vec[k, :]
            E_nk = D / B_vec[k] + v_vec[k] * np.dot(diff_nk, np.dot(W_k, diff_nk))
            P_nk = P_k - 0.5*E_nk
            P_vec[n, k] = P_nk
    return P_vec


def compute_responsibilities_R(P_vec):
    """
    Computes the responsibilities r_nk
    :param: P_vec [NxM]
    :return: R_vec [NxM]
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
        y_k *= 1.0 / N_k
        S_k *= 1.0 / N_k

        YHat_vec[k, :] = y_k
        S_vec[k*Y_vec.shape[1]:(k+1)*Y_vec.shape[1], :] = S_k
        #S = np.append(S, S_k)
    return YHat_vec, S_vec

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
    Computes the new m and Beta params of the Normal distribution.
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
    :return: v_vec [Mx1] array of updated v params for Whishart distrib of each GM k
    :return: W_vec Mx[D-by-D] stack of updated [DxD] covar matrices for each GM k
    """
    B_vec = np.zeros([R_vec.shape[1]]) # [Mx1]
    m_vec = np.zeros([R_vec.shape[1]], YHat_vec.shape[1]) # [MxD]
    v_vec = np.zeros([R_vec.shape[1]]) # [Mx1]
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




