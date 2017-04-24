import numpy as np
import re

def parser(file):

    mu_mu = get_mu_mu_from_file(file)
    mu_cov_list = get_mu_cov_from_file(file,mu_mu.shape[0])
    lambda_n = get_lambda_n_from_file(file)
    lambda_cov_list = get_lambda_cov_from_file(file,mu_mu.shape[0])
    alpha = get_alpha_from_file(file)

    return mu_mu, mu_cov_list,lambda_n,lambda_cov_list,alpha



def get_mu_mu_from_file(file):

    read_mu_mu = False

    with open(file, "r") as ins:
        
        # Mu_mu
        mu_list = []
        
        for line in ins:

            if 'Cov =' in line:

                read_mu_mu = True
                mu_mat = np.zeros([10 , 8])
                
                for i in range(len(mu_list)):
                    
                    mu_list[i] = re.sub(']', '', mu_list[i]) 
                    mu_list[i] = re.sub('  ', ' ', mu_list[i]) 
                    mu_list[i] = re.sub('   ', ' ', mu_list[i]) 
                    mu_mat[i,:] = np.fromstring(mu_list[i],dtype=float,sep =' ')
                return mu_mat
               
            if '[[' in line and read_mu_mu is False:
                mu_list += [line[2:-1]]

            elif '[' not in line and read_mu_mu is False and len(mu_list) > 0:
                mu_list[-1] += line[0:-2]


            elif '[' in line and read_mu_mu is False and len(mu_list) > 0:
                mu_list += [line[2:-1]]


def get_mu_cov_from_file(file,M):
    '''
    Inputs:
    ------
    file : path to text file containing the learnt parameters for a given locaiton
    M : number of mixands
    Outputs:
    -------
    mu_all_cov_list: list of 8x8 np.arrays, corresponding to each mixand component
    '''

    reading_mu_cov = False

    with open(file, "r") as ins:
        
        # Mu_mu
        
        mu_all_cov_list = []

        
        for line in ins:

            if 'Cov =' not in line and reading_mu_cov is False:
                continue
            elif reading_mu_cov is False:
                reading_mu_cov = True
                continue
            

            if '[[' in line :
                mu_cov_list = []
                mu_cov_list += [line.replace('[','')]

            elif '[' in line and len(mu_cov_list) > 0:
                mu_cov_list += [line.replace('[','')]

            elif ']]' in line:


                mu_cov = np.zeros([8 , 8])
                mu_cov_list[-1] += line.replace('[','')
                
                for i in range(len(mu_cov_list)):
                    mu_cov_list[i] = mu_cov_list[i].replace('  ', ' ') 
                    mu_cov_list[i] = mu_cov_list[i].replace('   ', ' ') 
                    mu_cov[i,:] = np.fromstring(mu_cov_list[i],dtype=float,sep =' ')
                mu_all_cov_list += [mu_cov]

                if len(mu_all_cov_list) == M:
                    break


            elif '[' not in line and len(mu_cov_list) > 0:
                mu_cov_list[-1] += line.replace('[','')

    return mu_all_cov_list  


def get_lambda_n_from_file(file):

    read_lambda_n = False

    with open(file, "r") as ins:
        
        # Mu_mu
        lambda_n_list = []
        
        for line in ins:
            
            if 'n =' in line:
                read_lambda_n = True
                
               
            elif '[' in line and read_lambda_n is True:

                lambda_n_list += [line.replace('[','')]

            elif ']' in line and read_lambda_n is True:
                
                lambda_n_list += [line.replace(']','')]
                lambda_n = np.zeros([2,5])
                
                for i in range(len(lambda_n_list)):
                    lambda_n[i,:] = np.fromstring(lambda_n_list[i],dtype=float,sep =' ')
                   
                return lambda_n.flatten()

def get_lambda_cov_from_file(file,M):
    '''
    Inputs:
    ------
    file : path to text file containing the learnt parameters for a given locaiton
    M : number of mixands
    Outputs:
    -------
    lambda_cov_all_list: list of 8x8 np.arrays, corresponding to each mixand component
    '''

    reading_lambda_cov = False

    with open(file, "r") as ins:
        
        # Mu_mu
        
        lambda_cov_all_list = []

        
        for line in ins:

            if 'A =' not in line and reading_lambda_cov is False:
                continue
            elif reading_lambda_cov is False:
                reading_lambda_cov = True
                continue
            

            if '[[' in line :
                Lambda_cov_list = []
                Lambda_cov_list += [line.replace('[','')]

            elif '[' in line and len(Lambda_cov_list) > 0:
                Lambda_cov_list += [line.replace('[','')]

            elif ']]' in line:


                lambda_cov = np.zeros([8 , 8])
                Lambda_cov_list[-1] += line.replace('[','')
                
                for i in range(len(Lambda_cov_list)):
                    Lambda_cov_list[i] = Lambda_cov_list[i].replace('  ', ' ') 
                    Lambda_cov_list[i] = Lambda_cov_list[i].replace('   ', ' ') 
                    lambda_cov[i,:] = np.fromstring(Lambda_cov_list[i],dtype=float,sep =' ')
                lambda_cov_all_list += [lambda_cov]

                if len(lambda_cov_all_list) == M:
                    break


            elif '[' not in line and len(Lambda_cov_list) > 0:
                Lambda_cov_list[-1] += line.replace('[','')

    return lambda_cov_all_list       

               
def get_alpha_from_file(file):

    read_alpha = False

    with open(file, "r") as ins:
        
        # Mu_mu
        alpha_list = []
        
        for line in ins:
            
            if 'alpha =' in line:
                read_alpha = True
                
               
            elif '[' in line and read_alpha is True:

                alpha_list += [line.replace('[','')]

            elif ']' in line and read_alpha is True:
                
                alpha_list += [line.replace(']','')]
                alpha = np.zeros([2,5])
                
                for i in range(len(alpha_list)):
                    alpha[i,:] = np.fromstring(alpha_list[i],dtype=float,sep =' ')
                   
                return alpha.flatten()
           






            
           

                

            

