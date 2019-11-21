from numpy import *
import numpy as np
import matplotlib as plt
import math

#隐马尔科链模型前向算法
def hmm_forward(A, PI, B, O):
    M = shape(PI)[0]   #观测序列大小
    N = shape(A)[1]    #状态序列大小
    T = M
    alpha = mat(zeros((M, N)))
    P = 0.0

    for i in range(N):
        alpha[0, i] = PI[i, 0] * B[i, 0]

    for t in range(T - 1):
        for i in range(N):
            temp_value = 0.0;
            for j in range(N):
                temp_value += alpha[t, j] * A[j, i]
            index = 0
            if(O[t + 1, 0] == 0):
                index = 0
            else:
                index = 1
            alpha[t + 1, i] = temp_value * B[i, index]
    for i in range(N):
        P += alpha[T - 1, i]
    return P,alpha

#隐马尔科链模型后向算法
def hmm_backword(A, PI, B, O):
    T,N = shape(A)
    beta = mat(zeros((T, N)))
    P = 0.0

    beta[T - 1, :] = 1
    t = T - 2
    while t >= 0:
        for i in range(N):
            temp_value = 0.0
            for j in range(N):
                index = 0
                if(O[t + 1, 0] == 0):
                    index = 0
                else:
                    index = 1
                temp_value += A[i, j] * B[j, index] * beta[t + 1, j]
            beta[t, i] = temp_value
        t -= 1

    for i in range(N):
        index = 0
        if(O[0, 0] == 0):
            index = 0
        else:
            index = 1
        P += PI[i, 0] * B[i, index] * beta[0, i]
    return P,beta

if __name__ == "__main__":
    A = mat([[0.5, 0.2, 0.2, 0.3],
             [0.3, 0.5, 0.2, 0.6],
             [0.2, 0.3, 0.5, 0.2],
             [0.5, 0.1, 0.2, 0.3]])
    B = mat([[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3],
             [0.2, 0.8]])
    PI = mat([[0.2],
              [0.4],
              [0.4],
              [0.3]])
    #红，白，红, 红
    O = mat([[0],
             [1],
             [0],
             [0]])

    P,alpha = hmm_forward(A, PI, B, O)#前向算法
    print(P)
    print("--------------------------------------")
    P,beta = hmm_backword(A, PI, B, O)#后向算法
    print(P)
