import numpy as np
from scipy import signal as sg
from numpy import linalg as lg


def compute_covariance(X, type='real'):
    r"""This function compute the covariance of a numpy matrix. 
        
        * If type='real', the covariance is computed as :math:`\textbf{R}=\frac{1}{N}\textbf{X}\textbf{X}^{T}`
        
        * If type='comp', the covariance is computed as :math:`\textbf{R}=\frac{1}{N}\textbf{X}\textbf{X}^{H}`
        
        :param X: M*N matrix
        :param type: string, optional
        :returns: covariance matrix of size M*M
        
        >>> import numpy as np
        >>> import spectral_analysis as sa
        >>> X = np.matrix('1 2; 3 4;5 6')
        >>> sa.compute_covariance(X)
        matrix([[  2.5,   5.5,   8.5],
        [  5.5,  12.5,  19.5],
        [  8.5,  19.5,  30.5]])
        
        """
        
    #Number of columns
    N=X.shape[1]
    if type=='comp':
        R=(1./N)*X*X.H
    else:
        R=(1./N)*X*X.T

    return R


def compute_autocovariance(x,M=-1):
    
    r""" This function compute the auto-covariance matrix of a numpy signal. The auto-covariance is computed as follows
        
        .. math:: \textbf{R}=\frac{1}{N}\sum_{M-1}^{N-1}\textbf{x}_{m}\textbf{x}_{m}^{T}
        
        where :math:`\textbf{x}_{m}^{T}=[x[m],x[m-1],x[m-M+1]]`.
        
    
        :param x: ndarray of size N
        :param M:  int, optional. Size of signal block. If M is equal to -1, then M=N/2.
        :returns: ndarray
        
        """
    
    # Create covariance matrix for psd estimation
    # length of the vector x
    N=x.shape[0]
    
    if M==-1:
        M=N/2
    
    # init covariance matrix
    yn=np.matrix(x[M-1::-1])
    R=yn.T*yn
    for indice in range(1,N-M):
        yn=np.matrix(x[M-1+indice:indice-1:-1])
        R=R+yn.T*yn
    
    R=R/N
    return R

def pseudospectrum_MUSIC(x,L,M,Fe,f_vect):
    
    #compute covariance matrix
    R=compute_autocovariance(x,M)

    #perform SVD of the covariance matrix
    U,S,V=lg.svd(R)

    #extract noise subspace
    U_noise=U[:,L:]

    #compute cost function MUSIC pseudo spectrum
    N_f=f_vect.shape
    cost=np.zeros(N_f)
    
    for indice,f_temp in enumerate(f_vect):
        # construct a
        vect_exp=2*np.pi*f_temp*np.arange(0,M)/Fe
        a=np.exp(-1j*vect_exp)
        a=np.transpose(np.matrix(a))
    
        cost[indice]=1./lg.norm((U_noise.H)*a)

    return cost

def root_MUSIC(x,L,M,Fe):

    #compute covariance matrix
    R=compute_covariance_psd(x,M)
    
    #perform SVD of the covariance matrix
    U,S,V=lg.svd(R)
    
    #extract noise subspace
    U_noise=U[:,L:]

    #construct polynomial