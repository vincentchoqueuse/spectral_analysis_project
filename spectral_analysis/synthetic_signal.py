import numpy as np
from scipy import signal as sg


def generate_sine(a0=1,f0=50,Fe=1000,N=1000):
    #time base
    t=1.*np.arange(N)/Fe
    #generate sine wave
    out=a0*np.sin(2*np.pi*f0*t)
    return (t,out)

def generate_saw(a0=1,f0=50,Fe=1000,N=1000,width=1):
    #time base
    t=1.*np.arange(N)/Fe
    #generate sawtooth wave
    out=a0*sg.sawtooth(2*np.pi*f0*t)
    return (t,out)

def generate_square(a0=1,f0=50,Fe=1000,N=1000,width=1):
    #time base
    t=1.*np.arange(N)/Fe
    #generate square wave
    out=a0*sg.square(2*np.pi*f0*t)
    return (t,out)

def generate_AM_PM_sine(a0=1,f0=50,ic=0.2,fc=1,pc=0,im=10,fm=1,pm=0,Fe=1000,N=1000):
    #time base
    t=1.*np.arange(N)/Fe
    #generate modulating signal
    am_signal=(1+ic*np.sin(2*np.pi*fc*t+pc))
    pm_signal=im*np.sin(2*np.pi*fm*t+pm)
    #generate modulated signal
    out=a0*am_signal*np.sin(2*np.pi*f0*t+pm_signal)
    return (t,out)

