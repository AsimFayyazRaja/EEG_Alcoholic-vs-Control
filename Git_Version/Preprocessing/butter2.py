from scipy.io import loadmat
import scipy
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle


#getting data
with open('features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)


eeg = features[0]
eeg1=eeg[0]    
fs = 256
fft1 = scipy.fft(eeg1)
f = np.linspace (0,fs,len(eeg1), endpoint=False)
plt.figure(1)
plt.plot (f, abs (fft1))
plt.title ('Magnitude spectrum of the signal')
plt.xlabel ('Frequency (Hz)')
plt.show()