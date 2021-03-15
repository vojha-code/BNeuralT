import os
import pandas as pd
import numpy as np

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#%% PART -I : MLP experiment results collection

cwd = os.getcwd()
print(cwd)
sfolder_Check = os.path.join(cwd,"trained_models"+os.sep+"traind_results_keras_TF_MLP")
resfolder_Check = os.listdir(sfolder_Check)
saveColl = os.path.join(cwd, "trained_models")
#%%
listData = []
listAlgo = []
listProblem = []
listAccTst = []
listPram = []
listTime = []

for fileName in  resfolder_Check:
    print(fileName)
    lststr = fileName.split('_')
    algoV = lststr[2]
    path_performance = os.path.join(sfolder_Check,fileName)
    read_np = np.load(path_performance).item()
    for keys in read_np:
        keyVal = read_np[keys]
        dataV = keys.split('_')[0]
        listData.append(dataV)
        #print(keys,dataV,read_np[keys])
        listAlgo.append(algoV)

        if len(keyVal) == 7:
            listProblem.append("reg")
            listAccTst.append(keyVal[3])
            listPram.append(keyVal[4])
            listTime.append(keyVal[5])
        else:
            listProblem.append("class")
            listAccTst.append(keyVal[1])
            listPram.append(keyVal[2])
            listTime.append(keyVal[3])
#%%
df_Table_MLP_coll = pd.DataFrame({'A' : []})
df_Table_MLP_coll['Problem'] = listProblem
df_Table_MLP_coll['Data'] = listData
df_Table_MLP_coll['Algo'] = listAlgo
df_Table_MLP_coll['AccTst'] = listAccTst
df_Table_MLP_coll['Param'] = listPram
#df_Table_MLP_coll['Time'] = listTime
del df_Table_MLP_coll['A']
#%%
df_Table_MLP_coll.to_csv(os.path.join(saveColl,'MLP_Keras_TF_models_coll.csv'))


#%% PART -II

dir_csv_coll = os.path.join(cwd,"trained_models")
file_name_coll = "BNeuralT_pre_trained_class_reg_models_coll.csv"
tablesFileName = os.path.join(dir_csv_coll,file_name_coll)

df_Table_BNeuralT = pd.read_csv(tablesFileName)
df_Table_BNeuralT.head()

data_name = [ 'australian',
               'heart',
               'ionosphere',
               'pima',
               'wdbc',
               'iris',
               'wine',
               'vehicle',
               'glass',
               'baseball',
               'dee',
               'diabetese',
               'friedman',
               'mpg6']


lisTAlgo = [
    "G_rmsprop_",
    "G_adagrad_",
    "G_nesterov_accelerated_gd_",
    "G_momentum_gd_",
    "G_gd_",
    "G_adam_",
    "G_Avg_Parm_W",
    "G_Best_Acc_G",
    "G_Best_G_W",
    "G_Algo_Best_of_G",
    "G_Time_tau",
    "M_rmsprop_",
    "M_adagrad_",
    "M_nesterov_accelerated_gd_",
    "M_momentum_gd_",
    "M_gd_",
    "M_adam_",
    "M_Parm_W",
    "M_Best_Acc_M",
    "M_Algo_Best_of_M",
    "M_Time_tau",
    ]

df_Table_Summary = pd.DataFrame({'Algo' : lisTAlgo})


for data in data_name:
    print(data)
    #BNeuralT results collection
    df_Data = df_Table_BNeuralT[df_Table_BNeuralT["Data"]==data]
    listRankAlgo = [1,5,4,3,2,0]
    df_DataMean = df_Data.groupby(["Algo"]).mean().reset_index()
    df_DataMean["Rank"] = listRankAlgo
    df_DataMean = df_DataMean.sort_values(["Rank"])
    df_DataMax = df_Data[df_Data["AccTst"] == np.max(df_Data["AccTst"])]
    lisTValue = []
    lisTValue = df_DataMean["AccTst"].tolist()
    lisTValue.append(df_DataMean["Param"].tolist()[0])
    lisTValue.append(np.max(df_DataMax["AccTst"].tolist()))
    lisTValue.append(np.min(df_DataMax["Param"].tolist()))
    lisTValue.append(df_DataMax["Algo"].tolist())
    lisTValue.append(np.mean(df_DataMean["FunEvalTime"].tolist()))

    #MLP results collection
    if(	data == "wdbc"):
        data = "wisconsin"
    df_Data = df_Table_MLP_coll[df_Table_MLP_coll["Data"]==data]
    listRankAlgo = [1,5,3,2,0,4]
    df_DataMean = df_Data.groupby(["Algo"]).mean().reset_index()
    df_DataMean["Rank"] = listRankAlgo
    df_DataMean = df_DataMean.sort_values(["Rank"])
    df_DataMax = df_Data[df_Data["AccTst"] == np.max(df_Data["AccTst"])]
    #lisTValue = []
    lisTValue.extend(df_DataMean["AccTst"].tolist())
    lisTValue.append(df_DataMean["Param"].tolist()[0])
    lisTValue.append(np.max(df_DataMax["AccTst"].tolist()))
    lisTValue.append(df_DataMax["Algo"].tolist())
    lisTValue.append("tau_script")
    df_Table_Summary[data] =  lisTValue

df_Table_Summary.to_csv(os.path.join(saveColl,'Table_1_class_reg_BNeuralT_MLP_models.csv'), index=False)

print("\nComplete: file saved")
print("Table_1_class_reg_BNeuralT_MLP_models.csv")
print("Location",cwd)
