import matplotlib.pyplot as plt
import numpy as np

def plot_sigmas_histories(histories, covar_type=None):
    
    # given a histories{} dict as given by fgnh.train()
    # plots the history of the inv_covariance and sigmas
    
    # get covar type if needed
    if covar_type==None:
        s = np.shape(histories['fl.inv_covars'])
        if len(s)==2:
            covar_type='sphere'
        elif len(s)==3:
            covar_type='diag'
        elif len(s)==4:
            # this could be full or cholesky, ask for clarification
            # covar_type='full'
             raise Exception('Is covar_type \'full\' or \'chol\'? ')

            
    # check that inv covar has gone up, sigmas down, and trace down if covar_type=='full'
    for k in histories.keys():
        if 'inv_covar' in k:
            s = histories[k].shape
            if covar_type == 'diag':
                # for each neuron
                inv_cov_2_plot = []
                sig_2_plot = []
                for i in range(len(histories[k][0])):
                    
                    inv_cov_2_plot.append(histories[k][:,i].reshape(s[0],np.prod(s[2:])))
#                     plt.plot(histories[k][:,i].reshape(s[0],np.prod(s[2:])), marker='.', linestyle=' ')
#                     plt.title('Inverse Covariance: '+k)
#                     plt.grid()
#                     plt.show()

                    sig_2_plot.append(1.0/histories[k][:,i].reshape(s[0],np.prod(s[2:])))
#                     plt.plot(1.0/histories[k][:,i].reshape(s[0],np.prod(s[2:])), marker='.', linestyle=' ')
#                     plt.title('Sigmas: '+k)
#                     plt.grid()
#                     plt.show()

                # plot a layer together
                for x in inv_cov_2_plot:
                    plt.plot(x)
                plt.title('Inverse Covariance: '+k)
                plt.grid()
                plt.show()
                for x in sig_2_plot:
                    plt.plot(x)
                plt.title('Sigmas: '+k)
                plt.grid()
                plt.show()
                                          

            elif covar_type == 'full':
                # for each neuron, plot trace of inv_covar
                for i in range(len(histories[k][0])):
                    # plot trace of inv_covar
                    trace = [np.einsum('ii->', p) for p in histories[k][:,i]]
                    plt.plot(trace, marker='.', linestyle=' ')
                    plt.title('Trace of Inverse Covariance Matrix: '+k)
                plt.grid()
                plt.show()
                
                # for each neuron, plot trace of sigma
                for i in range(len(histories[k][0])):
                    # plot trace of Sigmas by computing inverse stored inv_covar
                    trace = [np.einsum('ii->', np.linalg.inv(p)) for p in histories[k][:,i]]
                    plt.plot(trace, marker='.', linestyle=' ')
                    plt.title('Trace of Covariance Matrix: '+k)
                plt.grid()
                plt.show()
                
            elif covar_type == 'chol':
                # for each neuron, plot trace of inv_covar
                for i in range(len(histories[k][0])):
                    # plot trace of inv_covar
                    trace = [np.einsum('ii->', p) for p in histories[k][:,i]]
                    plt.plot(trace, marker='.', linestyle=' ')
                    plt.title('Trace of Inverse Covariance Half Matrix: '+k)
                plt.grid()
                plt.show()
                
                # for each neuron, plot trace of sigma
                for i in range(len(histories[k][0])):
                    # plot trace of Sigmas by computing inverse stored inv_covar
                    trace = [np.einsum('ii->', np.linalg.inv(np.matmul(p, np.transpose(p)))) for p in histories[k][:,i]]
                    plt.plot(trace, marker='.', linestyle=' ')
                    plt.title('Trace of Covariance Matrix: '+k)
                plt.grid()
                plt.show()
    

            elif covar_type == 'sphere':
                plt.plot(histories[k], marker='.', linestyle=' ',)
                plt.title('Inverse Covariance: '+k)
                plt.grid()
                plt.show()

                plt.plot(1.0/histories[k], marker='.', linestyle=' ')
                plt.title('Sigma :'+k)

                plt.grid()
                plt.show()
            
            else:
                # should not happen
                 raise Exception('Something went wrong with covar_type')
