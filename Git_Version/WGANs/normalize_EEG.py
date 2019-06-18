import glob
from tqdm import tqdm
import numpy as np
import os

import pickle
import matplotlib.pyplot as plt

#getting data
with open('../features', 'rb') as fp:
    features=pickle.load(fp)

with open('../labels', 'rb') as fp:
    labels=pickle.load(fp)

features=np.array(features)
labels=np.array(labels)

print(features.shape)
print(labels.shape)

ma=np.amax(features)
mi=np.amin(features)

#normalizing from -1 to 1

rang=ma-mi

feats=(features-mi)/rang
feats=feats*2
feats=feats-1


with open('features_normalized(-1,1)', 'wb') as fp:
    pickle.dump(features, fp)

with open('labels_normalized(-1,1)', 'wb') as fp:
    pickle.dump(labels, fp)

plt.plot(features[2002])
plt.title("Before normalization")
plt.savefig("norm.png")
plt.close()

plt.plot(feats[2002])
plt.title("After normalization")
plt.savefig("norm1.png")

