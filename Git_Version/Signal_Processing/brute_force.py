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
with open('updatedFeatures', 'rb') as fp:
    features=pickle.load(fp)

with open('../smoote_labels', 'rb') as fp:
    labels=pickle.load(fp)

svmtrain=[]
svmtest=[]
mntrain=[]
mntest=[]
rndtrain=[]
rndtest=[]

logrtrain=[]
logrtest=[]

from scipy.signal import argrelmax
from scipy.signal import find_peaks

height_count=1
width_count=1

info_file = open("Brute_Force_Results/brute_force_info_updated.txt", "w")
info_file.write('Iteration'+'\t'+'\t'+'Height'+'\t'+'\t'+'\t'+'Width'+'\t'+'\t'+'SVM'
    +'\t'+'\t'+'MNB'+'\t'+'\t'+'RndFor'+'\t'+'\t'+'LogReg')

info_file.write('\n')
info_file.write('\n')

count=0

m=20        #heights to try
n=6         #widths to try


svm_test=0
logr_test=0
rnd_test=0
mnb_test=0

for k in range(m):
    minlimit=height_count%18                              #controls min height limit
    maxlimit=np.random.randint(minlimit,minlimit+20)                 #controls max height limit
    width_count=0
    for l in range(n):    
        try:
            r=np.random.randint(1,6)
            minwidth=np.random.randint(0,4)                                   #controls min width
            maxwidth=np.random.randint(minwidth,minwidth+r)                      #controls max width

            svmt=0
            logrt=0
            rndt=0
            mnt=0

            train_features=[]
            i=0
            #gets data's num of peaks depending on height and width 
            with tqdm(total=len(features)) as pbar:
                for f in features:
                    sublist=[]
                    flag=False
                    sublist2=[]
                    sublist3=[]
                    for channel in f:
                        x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
                        width=[minwidth,maxwidth]))
                        x=x[0]
                        sublist3.append(len(x))     #training features for each X_train sample
                    train_features.append(np.array(sublist3))
                    pbar.update(1)
                    i+=1
            
            print("iteration: ", count)
            train_features=np.array(train_features)
            from sklearn import svm
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            y=[]
            for l in labels:
                if np.argmax(l)==0:    #alco
                    y.append(0)
                else:
                    y.append(1)     #control

            y=np.array(y)
            '''
            scaler = StandardScaler()
            scaler=scaler.fit(train_features)

            feats=copy.deepcopy(train_features)

            train_features=scaler.transform(train_features)
            '''
            feats=copy.deepcopy(train_features)
            X_train, X_test, y_train, y_test = train_test_split(train_features,y, test_size=0.1,shuffle=True)

            clf = svm.SVC(kernel='rbf' , C=0.3, max_iter=50000)
            clf.fit(X_train,y_train)

            svmt=clf.score(X_test,y_test)*100
            print("Testing acc=", str(svmt))
            svm_test=svmt
            from sklearn.naive_bayes import MultinomialNB

            clf = MultinomialNB()
            clf.fit(X_train,y_train)

            mnt=clf.score(X_test,y_test)*100
            print("Testing acc=", str(mnt))
            mnb_test=mnt
            from sklearn.ensemble import RandomForestClassifier

            
            clf = RandomForestClassifier(n_estimators=1200, random_state=0,n_jobs=8)
            clf.fit(X_train,y_train)

            rndt=clf.score(X_test,y_test)*100
            print("Testing acc=", str(rndt))
            rnd_test=rndt
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=8000,n_jobs=8)
            
            clf.fit(X_train,y_train)

            logrt=clf.score(X_test,y_test)*100
            print("Testing acc=", str(logrt))
            logr_test=logrt            
            info_file.write(str(count)+'\t'+'\t'+'\t'+'['+str(minlimit)+', '+str(maxlimit)
            +']'+'\t'+'\t'+'\t'+'['+str(minwidth)+', '+str(maxwidth)+']'+'\t'+'\t'+str(svm_test)
                +'\t'+'\t'+str(mnb_test)+'\t'+'\t'+str(rnd_test)+'\t'+'\t'+str(logr_test))
            info_file.write('\n')
            info_file.write('\n')
            
            count+=1
            svmtest.append(svmt)
            logrtest.append(logrt)
            mntest.append(mnt)
            rndtest.append(rndt)
            width_count+=1
        except Exception as e:
            print(e)
            count+=1
            svmtest.append(svmt)
            logrtest.append(logrt)
            mntest.append(mnt)
            rndtest.append(rndt)
        height_count+=1
        
try:
    r=np.arange(count)

    print(len(r))
    print(len(svmtest))

    p1=plt.scatter(r,svmtest, c='r',marker='o')
    p2=plt.scatter(r,mntest, c='g',marker='x')
    p3=plt.scatter(r,rndtest, c='y',marker='D')
    p4=plt.scatter(r,logrtest, c='k',marker='s')

    plt.legend((p1,p2,p3,p4),("SVM", "MultinomialNB","RNDFOREST", "LOGR"))
    plt.title("Peaks accuracies on different heights and widths")

    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracies')

    plt.savefig('Brute_Force_Results/results_updated.png')

    info_file.close()
except:
    info_file.close()
