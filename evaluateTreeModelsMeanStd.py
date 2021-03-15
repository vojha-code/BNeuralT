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



datfa_frames = []
for data in data_name:
    print(data)
    col_mu = data+"_mu"
    col_std = data+"_std"

    #BNeuralT results collection
    df_Data = df_Table_BNeuralT[df_Table_BNeuralT["Data"]==data]

    listRankAlgo = [1,5,4,3,2,0]
    df_DataMean = df_Data.groupby(["Algo"]).mean().reset_index()
    df_DataMean["Rank"] = listRankAlgo
    df_DataMean = df_DataMean.sort_values(["Rank"])
    df_DataMean = df_DataMean.set_index("Algo")

    df_DataMean[col_mu] = df_DataMean["AccTst"].tolist()
    df_DataMean = df_DataMean.transpose()
    df_DataMean = df_DataMean.reset_index(drop=False)

    df_DataStd = df_Data.groupby(["Algo"]).std().reset_index()
    df_DataStd["Rank"] = listRankAlgo
    df_DataStd = df_DataStd.sort_values(["Rank"])
    df_DataStd = df_DataStd.set_index("Algo")
    df_DataStd[col_std] = df_DataStd["AccTst"].tolist()
    df_DataStd = df_DataStd.transpose()
    df_DataStd = df_DataStd.reset_index(drop=False)

    frames = [df_DataMean, df_DataStd]
    df_Conct =pd.concat(frames)
    df_Conct = df_Conct[df_Conct["index"].isin([col_mu , col_std])]
    df_Conct = df_Conct.set_index(['index'])


    #MLP results collection
    if(	data == "wdbc"):
        data = "wisconsin"
    df_Data = df_Table_MLP_coll[df_Table_MLP_coll["Data"]==data]

    listRankAlgo = [1,5,3,2,0,4]
    df_DataMean = df_Data.groupby(["Algo"]).mean().reset_index()
    df_DataMean["Rank"] = listRankAlgo
    df_DataMean = df_DataMean.sort_values(["Rank"])
    df_DataMean = df_DataMean.set_index("Algo")
    df_DataMean[col_mu] = df_DataMean["AccTst"].tolist()
    df_DataMean = df_DataMean.transpose()
    df_DataMean = df_DataMean.reset_index(drop=False)

    df_DataStd = df_Data.groupby(["Algo"]).std().reset_index()
    df_DataStd["Rank"] = listRankAlgo
    df_DataStd = df_DataStd.sort_values(["Rank"])
    df_DataStd = df_DataStd.set_index("Algo")
    df_DataStd[col_std] = df_DataStd["AccTst"].tolist()
    df_DataStd = df_DataStd.transpose()
    df_DataStd = df_DataStd.reset_index(drop=False)

    frames = [df_DataMean, df_DataStd]
    df_Conct1 =pd.concat(frames)
    df_Conct1 = df_Conct1[df_Conct1["index"].isin([col_mu , col_std])].reset_index(drop=True)
    df_Conct1 = df_Conct1.set_index(['index'])


    df_Join =  pd.concat([df_Conct, df_Conct1], axis=1, sort=False)
    datfa_frames.append(df_Join)

df_Table_Summary = pd.concat(datfa_frames)


columnToName =  ["G_rmsprop_", "G_adagrad_", "G_nesterov_accelerated_gd_", "G_momentum_gd_", "G_gd_", "G_adam_", "M_RMSprop", "M_Adagrad", "M_NAG", "M_MGD", "M_SGD", "M_Adam"]
print(len(df_Table_Summary.columns), len(columnToName), df_Table_Summary.columns)
if (len(df_Table_Summary.columns) == len(columnToName)):
    df_Table_Summary.columns = columnToName

#%%
df_Table_Summary.to_csv(os.path.join(saveColl,'Table_1_mean_std_class_reg_BNeuralT_MLP_models.csv'))
print("\nComplete: file saved")
print("Table_1_mean_std_class_reg_BNeuralT_MLP_models.csv")
print("Location",cwd)
