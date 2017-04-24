import numpy as np
import plotting
import matplotlib.pyplot as plt


def load_log_likelihood_data(file_name, plt_color):
    log_likelihood = np.array([])
    with open(file_name) as f:
        for line in f:
            data = line.split()
            if len(data) == 1:
                log_likelihood = np.append(log_likelihood, float(data[0]))
        print 'L_vec = ', log_likelihood
        print 'Precision = ', np.abs(log_likelihood[-1] - log_likelihood[-2])
    split_name = file_name.split('_')[1]
    plotting.plot_lower_bound(log_likelihood, plt_color, split_name, False)


def log_like():
    dict = {'log_AVS':'dodgerblue', 'log_ORCCA':'magenta', 'log_corridorOffice':'lightgreen', 'log_corridorORCCA':'r'}
    for k, v in dict.items():
        load_log_likelihood_data(k, v)
    plt.show()

import os, inspect
def find_model_data_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    splitFolderName = 'code/'
    splitPath = path.split(splitFolderName)
    model_data_path = splitPath[0] + 'models_VB/'
    return model_data_path



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


file_name_list = ['AVS', 'ORCCA', 'corr_office', 'corr_ORCCA']
model_data_path = find_model_data_path()
string = 'alpha'
for file_name in file_name_list:
    param_vec = find_model_param(model_data_path+file_name, string)
    sum  = np.sum(param_vec)
    normalized_vec = param_vec * 1.0 / sum
    title = string +'_'+file_name
    print title + ' = ', normalized_vec, '\n'
    plt.figure()
    plt.title(title)
    plt.xlabel('GM k')
    plt.ylabel('alpha $\\alpha$')
    #plt.plot(normalized_vec, 'bo')
    plt.plot(param_vec, 'bo')
plt.show()


