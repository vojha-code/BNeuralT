# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:15:06 2020

@author: yl918888
"""

import os
import pandas as pd
import numpy as np
from random import randint

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#%%
sfolder_Check = os.path.join(r'')
resfolder_Check = os.listdir(sfolder_Check)
plotsPath = os.path.join(r"D:\BackUp_Research_GN\BPNT_Exp_Java\tables")
#%%

#exp	data	algo	error_trn	error_tst	corr_trn	corr_tst	r2_trn	r2_tst	tree_size	func_node	leaf_node	train_param	compute_time	problem

#%%
listExp = []
listData = []
listAlgo = []
listErrorTrn = []
listErrorTst = []

listR2Trn = []
listR2Tst = []

listPram = []
listTime = []
listProblem = []

listNull = [] 	


count = 0
for fileName in  resfolder_Check:
    lststr = fileName.replace('.', '_').split('B_1_')  
    
    if (
        lststr[1] == 'sigmoid_ES_No_Reg_No_npy' or 
        lststr[1] == 'sigmoid_ES_No_Reg_No_defopt_npy' or 
        lststr[1] == 'relu_ES_No_Reg_No_npy' or 
        lststr[1] == 'relu_ES_No_Reg_No_defopt_npy' or 
        lststr[1] == 'sigmoid_ES_No_Reg_l1_l2_npy' or 
        lststr[1] == 'sigmoid_ES_No_Reg_l1_l2_defopt_npy' or 
        lststr[1] == 'sigmoid_ES_50_Reg_No__npy' or 
        lststr[1] == 'sigmoid_ES_50_Reg_No_defopt__npy'
        ):
        print(fileName)
        
        count +=1
    

        path_performance = os.path.join(sfolder_Check,fileName)
        read_np = np.load(path_performance).item()
        for key in read_np:
            if(not (key == 'model_config')):
                keyStrList = key.split('_')
                dataV = keyStrList[0]
                algoV = keyStrList[2]
                keyVal = read_np[key]        
                
                listExp.append('MLP_'+lststr[1])
                listData.append(dataV)    
                listAlgo.append(algoV)
                
                if len(keyVal) == 7:
                    listProblem.append("reg")
                    
                    listErrorTrn.append(keyVal[0]) # keyVal[0] has mse trn
                    listErrorTst.append(keyVal[1]) # keyVal[1] has mse tst

                    listR2Trn.append(keyVal[2]) #  keyVal[2] has accuracy R2 trn
                    listR2Tst.append(keyVal[3]) #  keyVal[3] has accuracy R2 tst
                    
                    listPram.append(keyVal[4])  # trainable paramters
                    listTime.append(keyVal[5])  # compute time
                else:
                    listProblem.append("class")
                    
                    listErrorTrn.append(1.0 - keyVal[0]) # keyVal[0] has accuracy trn
                    listErrorTst.append(1.0 - keyVal[1]) # keyVal[1] has accuracy tst

                    listR2Trn.append(keyVal[0]) #  keyVal[0] has accuracy trn
                    listR2Tst.append(keyVal[1]) #  keyVal[1] has accuracy trn
                    
                    listPram.append(keyVal[2]) # trainable paramters
                    listTime.append(keyVal[3]) # compute time
                #common null
                listNull.append('') 
                
print('total expirments',(8*14),'=',count)
#%%
# exp	data	algo	
# error_trn	error_tst	corr_trn	corr_tst	r2_trn	r2_tst	
# tree_size	func_node	leaf_node	train_param	compute_time	
# problem


df_performance = pd.DataFrame({'A' : []})      

df_performance['exp'] = listExp
df_performance['data'] = listData
df_performance['algo'] = listAlgo
# performance
df_performance['error_trn'] = listErrorTrn
df_performance['error_tst'] = listErrorTst
df_performance['corr_trn'] = listNull # null
df_performance['corr_tst'] = listNull # null
df_performance['r2_trn'] = listR2Trn
df_performance['r2_tst'] = listR2Tst
# other params
df_performance['tree_size'] = listNull #null
df_performance['func_node'] = listNull # null
df_performance['leaf_node'] = listNull # null
df_performance['train_param'] = listPram
df_performance['compute_time'] = listTime
# problem
df_performance['problem'] = listProblem

del df_performance['A']
#%%◘
df_performance.to_csv(os.path.join(plotsPath,'1_MLP_Master_File.csv'),index=False)
print('data saved')
