import glob
from tqdm import tqdm
import numpy as np
import os
import copy
import pickle

files=os.listdir('new_data/')

i=0

c_list=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

c_present=set()

features=[]
labels=[]
m=0
channel_info=[]
with tqdm(total=len(files)) as pbar:
    for file in files:
        r=0
        for newfile in glob.glob("new_data/"+file+"/*.*"):
            f=open(newfile,'r')
            x=f.readline()
            x=x.split()
            x=x[1]
            lab=copy.deepcopy(x[3])
            j=0
            sublist=[]
            feature = []
            flag=False
            while(x):
                if flag:
                    flag=False
                    x = f.readline()
                    continue
                if j>4:
                    x=x.split()
                    channel=str(x[1])
                    if 'chan' in x or channel not in c_list:
                        j+=1
                        x=f.readline()
                        continue
                    try:
                        c_present.add(channel)
                        num=int(x[2])
                        value = float(x[3])
                        channel=str(x[1])
                        if num==255:
                            sublist.append(value)
                            feature.append(np.array(sublist))
                            sublist=[]
                            flag=True
                        else:
                            sublist.append(value)
                    except Exception as e:
                        print(e)
                        print(x)
                        pass
                x=f.readline()
                j+=1
            r+=1
            feature=np.array(feature)
            print("PRESENT IN DATA: ")
            print(c_present)

            print("DIFF: ")
            print(list(set(c_list) - c_present))
            
            if feature.shape ==(12,256):
                if lab=='a':
                    labels.append(np.array([1,0]))   #alcoholic
                elif lab=='c':
                    labels.append(np.array([0,1]))   #control
                features.append(feature)
            else:
                pass
                #print("passed")
        i+=1
        m+=1
        pbar.update(1)

features=np.array(features)
labels=np.array(labels)

print(features.shape)
print(labels.shape)

with open('12_features', 'wb') as fp:
    pickle.dump(features, fp)

with open('12_labels', 'wb') as fp:
    pickle.dump(labels, fp)

'''
k=0
all_features=[]
all_labels=[]
channel_info=[]
with tqdm(total=len(files)) as pbar:
    for file in files:
        r=0
        p=1
        features=[]
        labels=[]
        count=0
        for newfile in glob.glob("Raw_Data/"+file+"/*.*"):
            f=open(newfile,'r')
            x=f.readline()
            x=x.split()
            x=x[1]
            lab=copy.deepcopy(x[3])
            j=0
            sublist=[]
            feature = []
            flag=False
            count+=1
            while(x):
                if flag:
                    flag=False
                    x = f.readline()
                    continue
                if j>4:
                    x=x.split()
                    num=int(x[2])
                    channel=str(x[1])
                    value = float(x[3])
                    if num==255:
                        sublist.append(value)
                        feature.append(np.array(sublist))
                        sublist=[]
                        flag=True
                    else:
                        channel_info.append(channel)
                        sublist.append(value)
                x=f.readline()
                j+=1
            r+=1
            feature=np.array(feature)
            if feature.shape !=(64,256):
                print("passed")
                p=0
            else:
                features.append(feature)
            if lab=='a':
                labels=np.array([1,0])
            else:
                labels=np.array([0,1])
        features=np.array(features)
        print(features.shape)
        all_features.append(features)
        all_labels.append(np.array(labels))
        i+=1
        m+=1
        k+=1
        pbar.update(1)
'''
'''
labels = np.array(all_labels)
features = np.array(all_features)
channel_info=np.array(channel_info)
print(channel_info.shape)
print(labels.shape)
print(features.shape)


#print(features.transpose(0,2,1).shape)

#features=np.reshape(features,(len(features),16,16,64))

with open('features', 'wb') as fp:
    pickle.dump(features, fp)

with open('labels', 'wb') as fp:
    pickle.dump(labels, fp)
'''