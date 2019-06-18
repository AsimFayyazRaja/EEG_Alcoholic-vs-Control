from sklearn.manifold import TSNE
import pickle

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
import copy
from tqdm import tqdm
import _pickle as cPickle

with open('../Signal_Processing/LSTM_intermediate_features_peaks4', 'rb') as fp:
    features = pickle.load(fp)

'''
features=features.transpose(1,2,3,0)
features=np.reshape(features,(len(features),4,32,64))
print(features.shape)
'''


'''
print(features.shape)
print(labels.shape)
newlabels=[]
i=0
for l in labels:
    if np.argmax(l)==0:
        newlabels.append(0)
    else:
        newlabels.append(1)
    i+=1

newlabels=np.array(newlabels)

pca=PCA(n_components=4096)

features=np.reshape(features,(len(features),-1))

features=pca.fit_transform(features)

with open('../Manifold_features/pca_features', 'wb') as fp:
    pickle.dump(features, fp)

print(features.shape)
feats=fs.mutual_info_classif(features,newlabels,n_neighbors=7,random_state=0)

max_indices=sorted(range(len(feats)), key=lambda i: feats[i])[-256:]      #picking max 64 features
print(len(max_indices))


features=np.reshape(features,(len(features),-1))
newfeatures=[]
for f in features:
    newfeatures.append(f[max_indices])

features=np.array(newfeatures)
'''

'''
pca=PCA(n_components=512)

features=np.reshape(features,(len(features),-1))

features=pca.fit_transform(features)

with open('../Manifold_features/pca_features', 'wb') as fp:
    pickle.dump(features, fp)

'''

'''
with open('../Manifold_features/pca_features', 'rb') as fp:
    features=pickle.load(fp)
'''

features=np.reshape(features,(len(features),-1))

print(features.shape)

tsne = TSNE(n_components=3,perplexity=35, learning_rate=150,n_iter=50000
,n_iter_without_progress=200,)

X_embedded=tsne.fit_transform(features)

with open('../Manifold_features/tsne', 'wb') as fp:
    pickle.dump(X_embedded, fp)

print("KL-divergence: ", tsne.kl_divergence_)
print("Iterations: ",tsne.n_iter_)


'''

#save TSNE like this, after fitting
with open('tsne.pkl', 'wb') as fid:
    cPickle.dump(tsne, fid)

'''

'''
load it like this

with open('tsne.pkl', 'rb') as fid:
    tsne = cPickle.load(fid)

and then for any new feature F fo this:-

F=tsne.transform(F)		#returns F of size 3 after applying tsne on it
'''











