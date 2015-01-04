from spectral_analysis.synthetic_signal import generate_sine, generate_saw, generate_square,generate_AM_PM_sine
from spectral_analysis.spectral_analysis import pseudospectrum_MUSIC
from pylab import *


"""This is the main file
    """


Fe=1000
N=1000
t,out=generate_sine(1,53.20,Fe,N)


f_vect=np.arange(30,70,0.5)
print(f_vect)
cost=pseudospectrum_MUSIC(out,2,10,Fe,f_vect)
plot(f_vect,cost)

show()