#############################################
#   File: KNN_classification.py             #
#   Author: Li Yunzhe                       #
#   Contact: liyunzhe@whu.edu.cn            #
#   License: Copyright (c) 2019 Li Yunzhe   #
#############################################

import numpy as np
from math import sqrt
from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing

X_study = np.loadtxt('data_study.txt')#此处要进行np的import  import numpy as np
X_check = np.loadtxt('data_check.txt')
#获取标签,原来的标签是最后一行，并且是2和4，要改一下
labels = X_study[:,9]
labels = labels/2 -1
for i in range(len(labels)): 
    if labels[i] == 0: 
        labels[i] = 1
    else:
        labels[i] = 0
X_study = np.delete(X_study,9,axis=1)
label_check = X_check[:,9]
label_check = label_check/2 -1
for i in range(len(label_check)): 
    if label_check[i] == 0: 
        label_check[i] = 1
    else:
        label_check[i] = 0
X_check = np.delete(X_check,9,axis=1)
#print(labels)
total = list()

pca=PCA(n_components=3)
pca.fit(X_study)
x_study = pca.transform(X_study)

pca.fit(X_check)
x_check = pca.transform(X_check)

def distance(k, X_train, Y_train, x):
    assert 1 <= k <= X_train.shape[0], "K must be valid"
    assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], "the feature number of x must be equal to X_train"
    distance = [np.sum(abs(x_train - x)) for x_train in X_train]
    nearest = np.argsort(distance)
    topk_y = [Y_train[i] for i in nearest[:k]]
    votes = Counter(topk_y)
    return votes.most_common(1)[0][0]


if __name__ == "__main__":
    for xr in x_check:
    #x = np.array([8,4,5,1,2,1,7,3,1])
        label = distance(3, x_study, labels, xr)
        #print(label)
        
        total.append(label)
    print(total)
    
    result_NMI=metrics.normalized_mutual_info_score(label_check, total)
    print("result_NMI:",result_NMI)