#File: 代码.py                           
#Author: Li  Yunzhe                     
#Contact: liyunzhe@whu.edu.cn           
#License: Copyright (c) 2019 Li Yunzhe  

from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
from sklearn.model_selection import train_test_split
#导入数据
X = np.loadtxt('breast.txt')#此处要进行np的import  import numpy as np
#获取标签
label = X[:,9]
label = label/2 -1
for i in range(len(label)): 
    if label[i] == 0: 
        label[i] = 1
    else:
        label[i] = 0
X = np.delete(X,9,axis=1)
#数据标准化
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

#映射到均匀分布
#quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
#x_pac = quantile_transformer.fit_transform(X_scaled)

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pca.fit(X)
x = pca.transform(X_scaled)


#rbf_pca=KernelPCA(n_components=1,kernel='rbf',gamma=0.04)
#rbf_pca=KernelPCA(n_components=2,kernel='cosine',gamma=0.04)
#rbf_pca=KernelPCA(n_components=2,gamma=0.04)
#x=rbf_pca.fit_transform(x_pca)

#层次聚类
clustering = AgglomerativeClustering(n_clusters=2,linkage='ward').fit(x)

#clustering

#AgglomerativeClustering(n_clusters=2,linkage='ward')

#clustering.labels_

#kmean聚类
#kmeans = KMeans(n_clusters=2, random_state=0, max_iter=1000)#新建KMeans对象，并传入参数
#print(x)
#s = kmeans.fit(x)#进行训练
#print(s)
#print(kmeans.labels_)
#print(kmeans.predict([[0, 0], [4, 4]]))

#mean_shift聚类
##带宽，也就是以某个点为核心时的搜索半径
#bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=5000)
##设置均值偏移函数
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
##训练数据
#ms.fit(x)

#谱聚类
##变换成矩阵，输入必须是对称矩阵
#metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(x)).astype(np.int32)
#metrics_metrix += -1 * metrics_metrix.min()
##设置谱聚类函数
#n_clusters_= 2
#melabels = spectral_clustering(metrics_metrix,n_clusters=n_clusters_)
#print(melabels)
#print(kmeans.cluster_centers_)

print(clustering.labels_)

#result_NMI=metrics.normalized_mutual_info_score(label, kmeans.labels_)
result_NMI=metrics.normalized_mutual_info_score(label, clustering.labels_)
#result_NMI=metrics.normalized_mutual_info_score(label, ms.labels_)
#result_NMI=metrics.normalized_mutual_info_score(label, melabels)
print("result_NMI:",result_NMI)