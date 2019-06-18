import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from sklearn.decomposition import PCA

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#getting data
with open('ae_features_full', 'rb') as fp:
    features=pickle.load(fp)

with open('ae_labels_full', 'rb') as fp:
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

print(alco.shape)
print(control.shape)

alco_x=alco[:,0]
alco_y=alco[:,1]

print(alco_y)

alco_z=alco[:,2]
#p1=plt.scatter(alco_x,alco_y, c='r',marker='o')

control_x=control[:,0]
control_y=control[:,1]
control_z=control[:,2]
#p2=plt.scatter(control_x,control_y, c='b',marker='x')

'''
plt.legend((p1,p2),("Alcoholic", "Control"))
plt.title("AE results")
plt.show()
'''

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(alco_x, alco_y, alco_z, c='b',marker='x')

ax.scatter(control_x,control_y,control_z, c='r',marker='o')

pyplot.show()
