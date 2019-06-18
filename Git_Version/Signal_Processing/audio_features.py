from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm

from sklearn.model_selection import train_test_split

#getting data
with open('../smoote_features', 'rb') as fp:
    features=pickle.load(fp)

with open('../smoote_labels', 'rb') as fp:
    labels=pickle.load(fp)


X=[]

for f in features:
    

'''
from tsfresh import extract_features
extracted_features = extract_features(features)

print(len(extracted_features))

extracted_features=np.array(extracted_features)
print(extracted_features.shape)
'''