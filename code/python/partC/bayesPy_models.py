import os, inspect
import numpy as np
import plotting
import matplotlib.pyplot as plt
import parser_ben as parser

# ------------------------------------------ Log Likelihood ------------------------------------------#

def load_log_likelihood_data(file_name, plt_color):
    log_likelihood = np.array([])
    with open(file_name) as f:
        for line in f:
            data = line.split()
            if len(data) == 1:
                log_likelihood = np.append(log_likelihood, float(data[0]))
        print 'Location: ', file_name
        print 'L_vec = ', log_likelihood
        print 'Precision = ', np.abs(log_likelihood[-1] - log_likelihood[-2])
        print 'Iterations = ', len(log_likelihood)
        print '\n'
    split_name = file_name.split('_')[1]
    plotting.plot_lower_bound(log_likelihood, plt_color, split_name, False)

def show_results_log_like():
    dict = {'log_AVS':'dodgerblue', 'log_ORCCA':'magenta', 'log_corridorOffice':'lightgreen', 'log_corridorORCCA':'r'}
    for k, v in dict.items():
        load_log_likelihood_data(k, v)
    plt.show()

# ------------------------------------------ Access hyper-param parser ------------------------------------------#
def find_model_data_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    splitFolderName = 'code/'
    splitPath = path.split(splitFolderName)
    model_data_path = splitPath[0] + 'models_VB/'
    #print 'Loading model from: ', model_data_path
    return model_data_path


def get_param_init_values():
    alpha0 = 0.01
    B0 = 1.0 / 100.0
    return alpha0, B0

def load_model_data(file_name):
    model_data_path = find_model_data_path()
    mu_mu, mu_cov_list, lambda_n, lambda_cov_list, alpha = parser.parser(model_data_path + file_name)
    M = mu_mu.shape[0]
    D = mu_mu.shape[1]
    W_vec = np.zeros([M*D, D])
    k = 0
    for W in lambda_cov_list:
        W_vec[k*D:(k+1)*D, :] = W
        k += 1
    alpha0, B0 = get_param_init_values()
    B_vec = alpha.copy() - alpha0 + B0
    return alpha, B_vec, mu_mu, lambda_n, W_vec


# ------------------------------------------ Responsibilities ------------------------------------------#
def find_model_param(model_data_file, string):
    var_count = 0
    var_vec = np.array([])
    with open(model_data_file) as f:
        for line in f:
            if var_count > 1:
                data = line.split()
                for d in data:
                    try:
                        if d[-1]==']':
                            d = d[:-2]
                        var_vec = np.append(var_vec, float(d))
                    except:
                        continue
            if string in line:
                var_count += 1
    return var_vec

def show_results_responsibilities():
    dict = {'AVS':'dodgerblue', 'ORCCA':'magenta', 'corr_office':'lightgreen', 'corr_ORCCA':'r'}
    for file_name, color in dict.items():
        model_data_path = find_model_data_path()
        #print model_data_path + file_name
        alpha_vec = parser.get_alpha_from_file(model_data_path + file_name)
        #alpha_vec = find_model_param(model_data_path + file_name, 'alpha')
        print 'alpha_vec = ', alpha_vec
        alpha0, B0 = get_param_init_values()
        Nk_vec = alpha_vec.copy() - alpha0
        normalized_vec = Nk_vec * 1.0 / np.sum(Nk_vec)
        plotting.plot_responsibilities(normalized_vec, color, file_name, False)
    plt.show()




if __name__ == "__main__":
    #show_results_log_like()
    show_results_responsibilities()
