import numpy as np
from scipy import signal as sg
from numpy import linalg as lg


def compute_covariance(X):
    r"""This function estimate the covariance of a zero-mean numpy matrix. The covariance is estimated as :math:`\textbf{R}=\frac{1}{N}\textbf{X}\textbf{X}^{H}`
        
        
        :param X: M*N matrix
        :param type: string, optional
        :returns: covariance matrix of size M*M
        
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> X = np.matrix('1 2; 3 4;5 6')
        >>> sa.compute_covariance(X)
        matrix([[  2.5,   5.5,   8.5],
        [  5.5,  12.5,  19.5],
        [  8.5,  19.5,  30.5]])
        
        """
        
    #Number of columns
    N=X.shape[1]
    R=(1./N)*X*X.H

    return R


def compute_autocovariance(x,M):
    
    r""" This function compute the auto-covariance matrix of a numpy signal. The auto-covariance is computed as follows
        
        .. math:: \textbf{R}=\frac{1}{N}\sum_{M-1}^{N-1}\textbf{x}_{m}\textbf{x}_{m}^{H}
        
        where :math:`\textbf{x}_{m}^{T}=[x[m],x[m-1],x[m-M+1]]`.
        
        :param x: ndarray of size N
        :param M:  int, optional. Size of signal block.
        :returns: ndarray
        
        """
    
    # Create covariance matrix for psd estimation
    # length of the vector x
    N=x.shape[0]
    
    #Create column vector from row array
    x_vect=np.transpose(np.matrix(x))
    
    # init covariance matrix
    yn=x_vect[M-1::-1]
    R=yn*yn.H
    for indice in range(1,N-M):
        #extract the column vector
        yn=x_vect[M-1+indice:indice-1:-1]
        R=R+yn*yn.H
    
    R=R/N
    return R


def pseudospectrum_MUSIC(x,L,M=None,Fe=1,f=None):
    r""" This function compute the MUSIC pseudospectrum. The pseudo spectrum is defined as
        
        .. math:: S(f)=\frac{1}{\|\textbf{G}^{H}\textbf{a}(f) \|}
        
        where :math:`\textbf{G}` corresponds to the noise subspace and :math:`\textbf{a}(f)` is the steering vector. The peek locations give the frequencies of the signal.
        
        :param x: ndarray of size N
        :param L: int. Number of components to be extracted.
        :param M:  int, optional. Size of signal block.
        :param Fe: float. Sampling Frequency.
        :param f: nd array. Frequency locations f where the pseudo spectrum is evaluated.
        :returns: ndarray
        
        >>> from pylab import *
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fe=500
        >>> t=1.*np.arange(100)/Fe
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f,P=sa.pseudospectrum_MUSIC(x,1,100,Fe,None)
        >>> plot(f,P)
        >>> show()
        
        """
    
    # length of the vector x
    N=x.shape[0]
    
    if np.any(f)==None:
        f=np.linspace(0.,Fe//2,512)

    if M==None:
        M=N//2

    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #compute MUSIC pseudo spectrum
    N_f=f.shape
    cost=np.zeros(N_f)
    
    for indice,f_temp in enumerate(f):
        # construct a (note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T)
        vect_exp=-2j*np.pi*f_temp*np.arange(0,M)/Fe
        a=np.exp(vect_exp)
        a=np.transpose(np.matrix(a))
        #Cost function
        cost[indice]=1./lg.norm((G.H)*a)

    return f,cost

def root_MUSIC(x,L,M,Fe=1):
    
    r""" This function estimate the frequency components based on the roots MUSIC algorithm [BAR83]_ . The roots Music algorithm find the roots of the following polynomial
        
        .. math:: P(z)=\textbf{a}^{H}(z)\textbf{G}\textbf{G}^{H}\textbf{a}(z)
        
        The frequencies are related to the roots as 
        
        .. math:: z=e^{-2j\pi f/Fe}
        
        :param x: ndarray of size N
        :param L: int. Number of components to be extracted.
        :param M:  int, optional. Size of signal block.
        :param Fe: float. Sampling Frequency.
        :returns: ndarray containing the L frequencies
        
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fe=500
        >>> t=1.*np.arange(100)/Fe
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f=sa.root_MUSIC(x,1,None,Fe)
        >>> print(f)
        """

    # length of the vector x
    N=x.shape[0]
    
    if M==None:
        M=N//2
    
    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #construct matrix P
    P=G*G.H

    #construct polynomial Q
    Q=0j*np.zeros(2*M-1)
    #Extract the sum in each diagonal
    for (idx,val) in enumerate(range(M-1,-M,-1)):
        diag=np.diag(P,val)
        Q[idx]=np.sum(diag)

    #Compute the roots
    roots=np.roots(Q)

    #Keep the roots with radii <1 and with non zero imaginary part
    roots=np.extract(np.abs(roots)<1,roots)
    roots=np.extract(np.imag(roots) != 0,roots)

    #Find the L roots closest to the unit circle
    distance_from_circle=np.abs(np.abs(roots)-1)
    index_sort=np.argsort(distance_from_circle)
    component_roots=roots[index_sort[:L]]

    #extract frequencies ((note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T))
    angle=-np.angle(component_roots)

    #frequency normalisation
    f=Fe*angle/(2.*np.pi)

    return f

def Esprit(x,L,M,Fe):
    
    r""" This function estimate the frequency components based on the ESPRIT algorithm [ROY89]_ 
        
        The frequencies are related to the roots as :math:`z=e^{-2j\pi f/Fe}`. See [STO97]_ section 4.7 for more information about the implementation.
        
        :param x: ndarray of size N
        :param L: int. Number of components to be extracted.
        :param M:  int, optional. Size of signal block.
        :param Fe: float. Sampling Frequency.
        :returns: ndarray ndarray containing the L frequencies
        
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fe=500
        >>> t=1.*np.arange(100)/Fe
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f=sa.Esprit(x,1,None,Fe)
        >>> print(f)
        """

    # length of the vector x
    N=x.shape[0]
        
    if M==None:
        M=N//2

    #extract signal subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    S=U[:,:L]

    #Remove last row
    S1=S[:-1,:]
    #Remove first row
    S2=S[1:,:]

    #Compute matrix Phi (Stoica 4.7.12)
    Phi=(S1.H*S1).I*S1.H*S2

    #Perform eigenvalue decomposition
    V,U=lg.eig(Phi)

    #extract frequencies ((note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T))
    angle=-np.angle(V)
    
    #frequency normalisation
    f=Fe*angle/(2.*np.pi)
    
    return f