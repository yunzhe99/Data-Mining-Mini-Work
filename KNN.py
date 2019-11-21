from numpy import *
import operator
 
 
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
 
 
def kNN_Classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #关于tile函数的用法
    #>>> b=[1,3,5]
    #>>> tile(b,[2,3])
    #array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
    #       [1, 3, 5, 1, 3, 5, 1, 3, 5]])
    sqDiffMat = diffMat ** 2
    sqDistances = sum(sqDiffMat, axis = 1)
    distances = sqDistances ** 0.5#                           算距离
    sortedDistIndicies = argsort(distances)
    #关于argsort函数的用法
    #argsort函数返回的是数组值从小到大的索引值
    #>>> x = np.array([3, 1, 2])
    #>>> np.argsort(x)
    #array([1, 2, 0])
    classCount = {} #                                         定义一个字典
#   选择k个最近邻
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        #                                                     计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
 
    #                                                         返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex