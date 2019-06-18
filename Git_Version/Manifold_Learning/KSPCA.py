from sklearn.manifold import TSNE
import pickle

import _pickle as cPickle

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs


with open('../averaged_features', 'rb') as fp:
    features = pickle.load(fp)

features=np.reshape(features,(len(features),-1))
print(features.shape)

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler=scaler.fit(features)

features=scaler.transform(features)
'''

from sklearn.decomposition import KernelPCA

transformer = KernelPCA(n_components=1024, kernel='rbf',gamma=0.4,n_jobs=7)
X_embedded = transformer.fit_transform(features)

print(X_embedded.shape)

with open('../Manifold_features/KSPCA', 'wb') as fp:
    pickle.dump(X_embedded, fp)


