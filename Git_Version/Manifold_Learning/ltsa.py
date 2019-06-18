from sklearn.manifold import TSNE
import pickle

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
from sklearn.manifold import LocallyLinearEmbedding

with open('../features', 'rb') as fp:
    features1 = pickle.load(fp)

with open('../Manifold_features/pca_features', 'rb') as fp:
    features = pickle.load(fp)

with open('../labels', 'rb') as fp:
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

print(features.shape)
feats=fs.mutual_info_classif(features,newlabels,n_neighbors=5,random_state=0)

max_indices=sorted(range(len(feats)), key=lambda i: feats[i])[-64:]      #picking max 64 features
print(len(max_indices))


features=np.reshape(features,(len(features),-1))
newfeatures=[]
for f in features:
    newfeatures.append(f[max_indices])

features=np.array(newfeatures)
features=np.reshape(features,(len(features),-1))
print(features.shape)

lle=LocallyLinearEmbedding(n_components=2,max_iter=500,method='ltsa',n_jobs=7)

X_embedded=lle.fit_transform(features)

print(X_embedded.shape)

with open('../Manifold_features/ltsa', 'wb') as fp:
    pickle.dump(X_embedded, fp)