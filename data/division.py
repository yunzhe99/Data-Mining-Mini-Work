import numpy as np
import random

flag=np.zeros((700),dtype=int)
data_study=[]
data_check=[]
data_ori=np.loadtxt('breast.txt')

for i in range(175):
    num=random.randint(0,698)
    while(flag[num]==1):
        num=random.randint(0,698)
    data_study.append(data_ori[num])
    flag[num]=1

for i in range(699):
    if flag[i]==0:
        data_check.append(data_ori[i])

np.savetxt('data_study.txt',data_study,fmt='%.7e')
np.savetxt('data_check.txt',data_check,fmt='%.7e')
