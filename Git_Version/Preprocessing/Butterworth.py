from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz



fs = 256.0
lowcut = 0.1
highcut = 4.1

import pickle

with open('features', 'rb') as fp:
    features=pickle.load(fp)

data=features

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
FilteredData=[]
for i in range(len(data)):
    for j in range(64):
        temp=np.ones((64,256))
        y= butter_bandpass_filter(data[i][j], lowcut, highcut, fs, order=4)
        temp[j]=y
    FilteredData.append(temp)

update=np.array(FilteredData)
print(update.shape)

with open('butter_features', 'wb') as fp:
    pickle.dump(update, fp)