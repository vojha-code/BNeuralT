
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



lisTAlgo1 = [
    "G_rmsprop_",
    "G_adagrad_",
    "G_nesterov_accelerated_gd_",
    "G_momentum_gd_",
    "G_gd_",
    "G_adam_"]


lisTAlgo2 = ["M_Adagrad",
             "M_Adam",
             "M_MGD",
             "M_NAG",
             "M_RMSprop",
             "M_SGD"]



columns = ["data","Sample1","param"]
columns.extend(lisTAlgo1)
columns.extend(lisTAlgo2)
len(columns)

lisTAlgo1 = [
    "_rmsprop_",
    "_adagrad_",
    "_nesterov_accelerated_gd_",
    "_momentum_gd_",
    "_gd_",
    "_adam_"]


lisTAlgo2 = ["Adagrad",
             "Adam",
             "MGD",
             "NAG",
             "RMSprop",
             "SGD"]

df_stats = pd.DataFrame(columns=columns)
df_stats = pd.DataFrame(columns=columns)
from scipy.stats import ttest_ind
from scipy import stats
index = 0
for data in data_name:
    print(data)
    #BNeuralT results collection
    df_Data = df_Table_BNeuralT[df_Table_BNeuralT["Data"]==data]
    df_DataMean = df_Data.groupby(["Algo"]).mean().reset_index()
    algoMaxMean = df_DataMean[df_DataMean["AccTst"] == np.max(df_DataMean["AccTst"])].reset_index(drop=True)
    algoMaxMean = algoMaxMean["Algo"][0]
    # Select best algorithm
    cat1 = df_Data[df_Data['Algo'] == algoMaxMean].reset_index(drop=True)

    list_t_stats =[data, algoMaxMean, "t-stat"]
    list_p_val =[data, algoMaxMean, "p-val"]
    list_d_f =[data, algoMaxMean, "dof"]
    for alg in lisTAlgo1:
        if(not(alg == algoMaxMean)):
             #print(alg)
             cat2 = df_Data[df_Data['Algo'] == alg].reset_index(drop=True)
             ca1List = cat1['AccTst'].tolist()
             ca2List = cat2['AccTst'].tolist()
             #Alternative vaified
             #s_welch = np.sqrt(np.var(ca1List)/len(ca1List) + np.var(ca2List)/len(ca2List))
             #tval = (np.mean(ca1List)-np.mean(ca2List))/s_welch
             dof_numerator = (np.var(ca1List)/len(ca1List) + np.var(ca2List)/len(ca2List))**2
             dof_demoninator = ((np.var(ca1List)/len(ca1List))**2/(len(ca1List)-1)) + ((np.var(ca2List)/len(ca2List))**2/(len(ca2List)-1))
             dof = dof_numerator/dof_demoninator
             #t_score = stats.ttest_ind_from_stats(np.mean(ca1List), np.std(ca1List), len(ca1List), np.mean(ca2List), np.std(ca2List), len(ca1List), equal_var=False)
             res = ttest_ind(ca1List, ca2List, equal_var=False)
             list_t_stats.append(res[0])
             list_p_val.append(res[1]/2)
             list_d_f.append(round(dof))
             #print(' ',alg, res[0], res[1],  dof)
        else:
            list_t_stats.append("")
            list_p_val.append("")
            list_d_f.append("")

    #MLP results collection
    if(	data == "wdbc"):
        data = "wisconsin"

    df_Data = df_Table_MLP_coll[df_Table_MLP_coll["Data"]==data]
    for alg in lisTAlgo2:
        #print(alg)
        cat2 = df_Data[df_Data['Algo'] == alg].reset_index(drop=True)
        ca1List = cat1['AccTst'].tolist()
        ca2List = cat2['AccTst'].tolist()
        #Alternative vaified
        #s_welch = np.sqrt(np.var(ca1List)/len(ca1List) + np.var(ca2List)/len(ca2List))
        #tval = (np.mean(ca1List)-np.mean(ca2List))/s_welch
        dof_numerator = (np.var(ca1List)/len(ca1List) + np.var(ca2List)/len(ca2List))**2
        dof_demoninator = ((np.var(ca1List)/len(ca1List))**2/(len(ca1List)-1)) + ((np.var(ca2List)/len(ca2List))**2/(len(ca2List)-1))
        dof = dof_numerator/dof_demoninator
        #t_score = stats.ttest_ind_from_stats(np.mean(ca1List), np.std(ca1List), len(ca1List), np.mean(ca2List), np.std(ca2List), len(ca1List), equal_var=False)
        res = ttest_ind(ca1List, ca2List, equal_var=False)
        list_t_stats.append(res[0])
        list_p_val.append(res[1]/2)
        list_d_f.append(round(dof))
        #print(' ',alg, res[0], res[1],  dof)


    df_stats.loc[index] = list_t_stats
    index += 1
    df_stats.loc[index] = list_p_val
    index += 1
    df_stats.loc[index] = list_d_f
    index += 1


#%%
df_stats.to_csv(os.path.join(saveColl,'Table_1_stats_class_reg_BNeuralT_MLP_models.csv'), index=False)

print("\nComplete: file saved")
print("Table_1_stats_class_reg_BNeuralT_MLP_models.csv")
print("Location",cwd)
