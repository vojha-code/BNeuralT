'''

Anonymous Author(s)

'''
import os
import pandas as pd
import numpy as np
from random import randint

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#%%
sfolder_Check = os.path.join(r'outputs')
resfolder_Check = os.listdir(sfolder_Check)
summaryPath = os.path.join(r'outputs')
#%%
strT = 'keras_iris_Ep_10_B50D_sig_all.npy'
lststr = strT.split('_')
data = lststr[1]


path_performance = os.path.join(sfolder_Check,resfolder_Check[0])
read_np = np.load(path_performance).item()
class_list = read_np['iris_1_Adagrad']
len(class_list)

reg_list = read_np['dee_1_Adagrad']
len(reg_list)

kel_list = 'dee_1_Adagrad'.split('_')

#%%
listData = []
listAlgo = []
listProblem = []
listErrorTrn = []
listErrorTst = []
listR2Trn = []
listR2Tst = []
listPram = []
listTime = []

for fileName in  resfolder_Check:
    print(fileName)
    lststr = fileName.split('_')
    
    path_performance = os.path.join(sfolder_Check,fileName)
    read_np = np.load(path_performance).item()
    for keys in read_np:
        keyVal = read_np[keys]
        dataV = keys.split('_')[0]
        algoV = keys.split('_')[2]
        listData.append(dataV)    
        #print(keys,dataV,read_np[keys])
        listAlgo.append(algoV)
        
        listErrorTrn.append(keyVal[0])
        listErrorTst.append(keyVal[1])
        if len(keyVal) == 7:
            listProblem.append("reg")
            listR2Trn.append(keyVal[2])
            listR2Tst.append(keyVal[3])
            listPram.append(keyVal[4])
            listTime.append(keyVal[5])
        else:
            listProblem.append("class")
            listR2Trn.append(99999)
            listR2Tst.append(99999)
            listPram.append(keyVal[2])
            listTime.append(keyVal[3])
#%%
df_performance = pd.DataFrame({'A' : []})      
df_performance['problem'] = listProblem
df_performance['data'] = listData
df_performance['algo'] = listAlgo
df_performance['error_trn'] = listErrorTrn
df_performance['error_tst'] = listErrorTst
df_performance['r2_trn'] = listR2Trn
df_performance['r2_tst'] = listR2Tst
df_performance['param'] = listPram
df_performance['time'] = listTime
del df_performance['A']
#%%
df_performance.to_csv(os.path.join(summaryPath,'MLP_TF_performance.csv'))
