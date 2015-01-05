from spectral_analysis.synthetic_signal import generate_sine, generate_saw, generate_square,generate_AM_PM_sine
import spectral_analysis.spectral_analysis as sa
from pylab import *
import numpy as np



Fe=500
t=1.*np.arange(100)/Fe
x=np.exp(2j*np.pi*55.2*t)

f_root_MUSIC=sa.root_MUSIC(x,1,None,Fe)
print('Frequencies Extracted with root_MUSIC (Hz)')
print(f_root_MUSIC)
f_Esprit=sa.Esprit(x,2,None,Fe)
print('Frequencies Extracted with ESPRIT (Hz):')
print(f_Esprit)