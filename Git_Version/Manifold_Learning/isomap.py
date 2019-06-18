from sklearn.manifold import TSNE
import pickle

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
from sklearn.manifold import Isomap

with open('../Conv_perchannel/perchannel_conv_features', 'rb') as fp:
    features = pickle.load(fp)

with open('../averaged_labels', 'rb') as fp:
    labels = pickle.load(fp)

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

features=features.transpose(1,2,3,0)
features=np.reshape(features,(len(features),4,32,64))
print(features.shape)

'''
feats=fs.mutual_info_classif(features,newlabels,n_neighbors=5,random_state=0)

max_indices=sorted(range(len(feats)), key=lambda i: feats[i])[-64:]      #picking max 64 features
print(len(max_indices))


features=np.reshape(features,(len(features),-1))
newfeatures=[]
for f in features:
    newfeatures.append(f[max_indices])

features=np.array(newfeatures)
'''
features=np.reshape(features,(len(features),-1))
print(features.shape)

lle=Isomap(n_components=10,max_iter=60000,n_jobs=-1)

X_embedded=lle.fit_transform(features)

print(X_embedded.shape)

with open('../Manifold_features/isomap', 'wb') as fp:
    pickle.dump(X_embedded, fp)