import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from sklearn.decomposition import PCA

#getting data
with open('../Manifold_features/ltsa', 'rb') as fp:
    features=pickle.load(fp)

with open('../averaged_labels', 'rb') as fp:
    labels=pickle.load(fp)

alco=[]
control=[]
i=0
print(features.shape)
print(labels.shape)
for l in labels:
    if np.argmax(l)==0:
        alco.append(features[i])
    else:
        control.append(features[i])
    i+=1

alco=np.array(alco)
control=np.array(control)

alco_x=alco[:,0]
alco_y=alco[:,1]
'''
alco_x=alco_x[:50]
alco_y=alco_y[:50]
'''
p1=plt.scatter(alco_x,alco_y, c='r',marker='o')

control_x=control[:,0]
control_y=control[:,1]
'''
control_x=control_x[:50]
control_y=control_y[:50]
'''
p2=plt.scatter(control_x,control_y, c='b',marker='x')
plt.legend((p1,p2),("Alcoholic", "Control"))
plt.title("LTSA results")
plt.show()