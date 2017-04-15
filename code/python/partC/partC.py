from scipy import io


def load_matlab_data(file, variable):
    mat = io.loadmat(file)
    data = mat[variable]
    return data

X_bar = load_matlab_data('testIsomapGMtraining_Xalldata', 'Xall') # [2141 x 3] ?
Y_bar = load_matlab_data('YbarData', 'Ybar') # [8 x 6000]

#print 'X_bar = ', X_bar
#print 'Y_bar = ', Y_bar
