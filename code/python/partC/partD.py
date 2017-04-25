import os, inspect
import numpy as np
from numpy import linalg as la
from scipy.special import gamma
import bayesPy_models as models
from scipy.io import loadmat


class Location_Model():
    def __init__(self, model_tag, alpha_vec, B_vec, m_vec, v_vec, W_vec):#, priors=5*[None]):
        self.model_tag = model_tag
        self.alpha_vec = alpha_vec
        self.B_vec = B_vec
        self.m_vec = m_vec
        self.v_vec = v_vec
        self.W_vec = W_vec

# model_data_dict = {'AVS': LM_AVS, 'ORCCA': LM_ORCCA, 'corr_office': LM_corridor_office, 'corr_ORCCA': LM_corridor_orcca}
def compute_classification_result_L_star(Y_l_eval, model_data_dict):
    """
    Computes the classification result L* over all possible l places
    :param: Y_l_eval [P x D] where P = 3072
    :param: model_data_list: list with {model_tag: Location_Model} for all locations: AVS, ORCCA, corr_office, corr_ORCCA
    :return :  L*
    """
    L_star_vec = np.zeros(len(model_data_list))
    l = 0
    for model_tag, model in model_data_dict.items():
        print 'LM = ', model_tag
        L_star_vec[l] = compute_model_log_likelihood(Y_l_eval, model)
        l += 1
    L_star = np.argmax(L_star_vec)
    print 'L_star_vec = ', L_star_vec
    print 'L_star = ', L_star
    return L_star


def compute_model_log_likelihood(Y_l_eval, loc_model):
    """
    Computes the log-likelihood for all essential component features y_l_r_p that make up the image I_l_r
    :param: Y_l_eval [P x D] where P = 3072
    :param: loc_model: Location_Model object for location L=l
    :return: sum_log_p = sum_p (log P(y_l_r_p))
    """
    P = Y_l_eval.shape[0]
    sum_log_p = 0.0
    for p in range(0, P):
        p_y_predicted = compute_predictive_density(Y_l_eval[p, :], loc_model.alpha_vec, loc_model.B_vec,
                                                   loc_model.m_vec, loc_model.v_vec, loc_model.W_vec)
        sum_log_p += np.log(p_y_predicted) 
    return sum_log_p

def compute_predictive_density(y_l_r_p, alpha_vec, B_vec, m_vec, v_vec, W_vec):
    """
    Computes the predictive density for a new data point y_n: p_y_predicted
    :param: y_l_r_m [Dx1]  feature vector
    :param: alpha_vec [Mx1] alpha parameters for the Dirichlet
    :param: B_vec [Mx1] Beta params for the Normal
    :param: m_vec [MxD] [Dx1] m params for the Normal
    :param: v_vec [Mx1] v params for Wishart
    :param: W_vec Mx[D-by-D] stack of [DxD] covar matrices for Wishart
    :return: p_y_predicted
    """
    D = len(y_l_r_p)
    alpha_hat = np.sum(alpha_vec)
    sum_k = 0.0
    for k in range(0, len(alpha_vec)):
        nu = v_vec[k] + 1.0 - D # scalar
        L = W_vec[k*D:(k+1)*D, :] * B_vec[k] * nu / (1.0 + B_vec[k])
        St = compute_pdf_multivariate_student_T(y_l_r_p, m_vec[k, :], L, nu)
        sum_k += alpha_vec[k] * St
    p_y_predicted = (1.0 / alpha_hat) * sum_k
    return p_y_predicted


def compute_pdf_multivariate_student_T(y, mu, lam, nu):
    """
    Computes the Student's T pdf of a multivariate distribution, St
    :param: y [Dx1]
    :param: mu [Dx1] mean
    :param: lam [DxD] covariance
    :param: nu [scalar] # DOF
    :return: St = aa * bb * cc
    """
    D = len(y)
    aa = gamma(0.5*nu + 0.5*D) / gamma(0.5*nu)
    bb = np.sqrt(la.det(lam)) / np.power(nu*np.pi, 0.5*D)
    pow = -0.5*(nu + D)
    cc = np.power(1.0 + (1.0 / nu) * np.dot(y - mu, np.dot(lam, y - mu)), pow)
    St = aa * bb * cc
    print 'aa = ', aa
    print 'bb = ', bb
    print 'cc = ', cc
    return St



def create_loc_model(model_tag):
    alpha_vec, B_vec, m_vec, v_vec, W_vec = models.load_model_data(model_tag)
    LM = Location_Model(model_tag, alpha_vec, B_vec, m_vec, v_vec, W_vec)
    return LM


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

def load_evaluation_sets(folder_name):
    matPath = find_matlab_data_path()
    mat_folder_name = matPath + folder_name
    mats = []
    for file in os.listdir(mat_folder_name):
        mats.append(loadmat(mat_folder_name + file)['y'].T)
    return mats
# ------------------------------------------ MAIN ------------------------------------------#
if __name__ == "__main__":
    model_data_list = ['AVS', 'ORCCA', 'corr_office', 'corr_ORCCA']
    LM_dict = {}
    #file_name_list = ['ORCCA']
    for model_name in model_data_list:
        LM = create_loc_model(model_name)
        LM_dict[model_name] = LM
        def print_location_model():
            print model_name,
            print 'alpha_vec = ', LM.alpha_vec
            print 'B_vec = ', LM.B_vec
            print 'v_vec = ', LM.v_vec
            print 'm_vec = ', LM.m_vec.shape
            print 'W_vec = ', LM.W_vec.shape
        #print_location_model()

    folder_name = 'Y_hat_orcca/'
    Y_l_eval_list = load_evaluation_sets(folder_name)
    print 'Testing data: ', folder_name
    image_idx = 0
    for Y_l_eval in Y_l_eval_list:
        print 'Image number: ', image_idx
        compute_classification_result_L_star(Y_l_eval, LM_dict)
        image_idx += 1

