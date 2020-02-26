import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
import random
from time import time

def build(G, n, HyG):
    '''
    在构建超图的过程中调用，是单次模拟过程
    '''
    activated_nodes=[]
    activated_nodes.append(n)
    for node in activated_nodes:
        fathers=G.predecessors(node)
        for father in fathers:
            prob=random.random()
            if prob<G[father][node]['weight']:
                if father not in activated_nodes:
                    activated_nodes.append(father)
    #print(activated_nodes)
    HyG.append(activated_nodes[1:])

# 构建超图
def BuildHypergraph(R,G):
    '''
    构建一个超图，这个超图是一个稠密图，通过反向模拟来获取尽可能多的边的信息，反向模拟R次
    '''
    HyG=[]

    for i in range(R):
        # choose node u from G uniformly at random
        u=random.randint(1,200)
        #print(u)
        build(G,u,HyG)
    #print(HyG)
    return HyG

# 找连接最多的节点
def select_max(H,flag):
    '''
    寻找最大的节点并进行将找到的删除，是找候选集的子操作
    '''
    for i in H:
        for j in i:
            flag[j]=flag[j]+1
    max_node=flag.index(max(flag))

    for j in H:
        if max_node in j:
            H.remove(j)

    return max_node

# 找候选集
def BuildSeedSet(H, k):
    '''
    找到k个候选集，选择标准是超图中节点的度数，即将反向模拟过程中模拟到次数最多的点选出
    '''
    Seed=[]

    for i in range(k):
        flag=[0]*201
        # select k seed node
        node=select_max(H,flag)
        Seed.append(node)
    return Seed

if __name__ == '__main__':
    # 加载图的数据
    tic = time()
    graph_data=np.matrix((np.loadtxt('graph.txt')))
    #graph=nx.DiGraph()
    #graph.add_edges_from(graph_data)
    #print (graph_data.shape)
    graph=nx.DiGraph()
    #print(graph)
    for i in range(1,201):
        for j in range(1,201):
            if graph_data[i,j]>0:
                graph.add_edge(i,j,weight=graph_data[i,j])
    #print(graph.edges(data=True))
    #nx.draw(graph, with_labels=True)
    #print(graph)
    #plt.show()

    # 主过程
    hpgraph=BuildHypergraph(1500,graph)

    seednodes=BuildSeedSet(hpgraph, 10)
    print("程序用时", (time() - tic), "s")
    print(seednodes)
