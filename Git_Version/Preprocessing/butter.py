from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy import signal
import copy

#getting data
with open('features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

print(np.unique(labels))
'''

features=features[:1280]
features=np.reshape(features,(len(features),-1))


print(features.shape)

from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 256.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(features, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
'''


import numpy as np

fs = 256
data=copy.deepcopy(features)
#print(max(data[0][0]))

fft_vals=[]
for d in data:
    fft_vals=np.absolute(np.fft.fft2(d))
    print(fft_vals.shape)
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.fftfreq(len(d), 1.0/fs)
    print(fft_freq.shape)

    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                'Theta': (4, 8),
                'Alpha': (8, 12),
                'Beta': (12, 30),
                'Gamma': (30, 45)}

    delta=[]
    theta=[]
    alpha=[]
    beta=[]
    gamma=[]

    delta_labels=[]
    theta_labels=[]
    alpha_labels=[]
    beta_labels=[]
    gamma_labels=[]

    delta_real=[]
    theta_real=[]
    alpha_real=[]
    beta_real=[]
    gamma_real=[]

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                        (fft_freq <= eeg_bands[band][1]))[0]
        if band=='Delta':
            delta_real=data[freq_ix]
            delta_labels=labels[freq_ix]
            print(delta_labels.shape)
            delta=fft_vals[freq_ix]
        elif band=='Theta':
            theta_real=data[freq_ix]
            theta_labels=labels[freq_ix]
            theta=fft_vals[freq_ix]
        elif band=='Alpha':
            alpha_real=data[freq_ix]
            alpha_labels=labels[freq_ix]
            alpha=fft_vals[freq_ix]
        elif band=='Beta':
            beta_real=data[freq_ix]
            beta_labels=labels[freq_ix]
            beta=fft_vals[freq_ix]
        elif band=='Gamma':
            gamma_real=data[freq_ix]
            gamma_labels=labels[freq_ix]
            gamma=fft_vals[freq_ix]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])


    print(delta.shape)
    print(theta.shape)
    print(alpha.shape)
    print(beta.shape)
    print(gamma.shape)

    print(delta_labels.shape)
    print(theta_labels.shape)
    print(alpha_labels.shape)
    print(beta_labels.shape)
    print(gamma_labels.shape)

    print(delta_real.shape)
    print(theta_real.shape)
    print(alpha_real.shape)
    print(beta_real.shape)
    print(gamma_real.shape)

'''
with open('./FreqDomains/delta_FreqDomain', 'wb') as fp:
    pickle.dump(delta, fp)

with open('./FreqDomains/theta_FreqDomain', 'wb') as fp:
    pickle.dump(theta, fp)

with open('./FreqDomains/alpha_FreqDomain', 'wb') as fp:
    pickle.dump(alpha, fp)

with open('./FreqDomains/beta_FreqDomain', 'wb') as fp:
    pickle.dump(beta, fp)

with open('./FreqDomains/gamma_FreqDomain', 'wb') as fp:
    pickle.dump(gamma, fp)


with open('./FreqDomains/delta_labels', 'wb') as fp:
    pickle.dump(delta_labels, fp)

with open('./FreqDomains/theta_labels', 'wb') as fp:
    pickle.dump(theta_labels, fp)

with open('./FreqDomains/alpha_labels', 'wb') as fp:
    pickle.dump(alpha_labels, fp)

with open('./FreqDomains/beta_labels', 'wb') as fp:
    pickle.dump(beta_labels, fp)

with open('./FreqDomains/gamma_labels', 'wb') as fp:
    pickle.dump(gamma_labels, fp)



with open('./TimeDomains/delta_TimeDomain', 'wb') as fp:
    pickle.dump(delta_real, fp)

with open('./TimeDomains/theta_TimeDomain', 'wb') as fp:
    pickle.dump(theta_real, fp)

with open('./TimeDomains/alpha_TimeDomain', 'wb') as fp:
    pickle.dump(alpha_real, fp)

with open('./TimeDomains/beta_TimeDomain', 'wb') as fp:
    pickle.dump(beta_real, fp)

with open('./TimeDomains/gamma_TimeDomain', 'wb') as fp:
    pickle.dump(gamma_real, fp)


with open('./FreqDomains/delta_labels', 'wb') as fp:
    pickle.dump(delta_labels, fp)

with open('./FreqDomains/theta_labels', 'wb') as fp:
    pickle.dump(theta_labels, fp)

with open('./FreqDomains/alpha_labels', 'wb') as fp:
    pickle.dump(alpha_labels, fp)

with open('./FreqDomains/beta_labels', 'wb') as fp:
    pickle.dump(beta_labels, fp)

with open('./FreqDomains/gamma_labels', 'wb') as fp:
    pickle.dump(gamma_labels, fp)
'''

#np.fft.ifft2()

'''
delta=np.reshape(delta,(len(delta),256,64))
plt.plot(delta[0])
plt.title("Delta Waves")
plt.show()
'''



'''
# Plot the data (using pandas here cause it's easy)
import pandas as pd
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]

print(eeg_band_fft)

ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
plt.show()
'''
