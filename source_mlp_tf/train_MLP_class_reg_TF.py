'''

Anonymous Author(s)

'''
from data_retrival import SettingParamter
from data_loader import data_evaluation
from data_loader import data_partition
from data_processing import DataProcessing
import numpy as np
import os
from datetime import datetime
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf
from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.regularizers import l1
from keras.regularizers import l1_l2

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
#%%Clasification
#DATA PREPARATION
data_file_set = [
                'iris',     # 0 
                'wisconsin',# 1
                'wine',     # 2
                'heart.csv',# 3
                'ionosphere.csv',# 4
                'australian.csv',# 5
                'pima.csv',   # 6 
                'glass.csv',  # 7
                'vehicle.csv',# 8 
                #
                'dee.csv',    # 9 
                'diabetese.csv',# 10 
                'baseball.csv', # 11 
                'mpg6.csv',     # 12 
                'friedman.csv'] # 13



set_running = 8

if (set_running == 1):
    n_problem_type = 'Classification' # 'Regression' 9 on  # 'Classification'
    data_file_set = ['iris']


n_problem_type = 'Classification'# 'Regression' 9 on  # 'Classification'
data_file_set = [data_file_set[6]]    

print('\n running : ',data_file_set,'\n')      



solver=['RMSprop', 'Adam','Adagrad', 'SGD','MGD', 'NAG']

EPOCHS=50 # 500 
BATCH_SIZE = 1 # 1 #hgfj
EXP_RUN = 1 # 30

FUN = 'sigmoid'
ES = 5 # Percentage of Epochs
ES = int(ES/100 * EPOCHS)
# ES = 'No'
REG = 'No' # 'l1_l2'
OptSet = ['_','_defopt']
#%%
for runOpt in OptSet:
    data_results = []
    for j in range(len(data_file_set)):
        #if j < 9:
        #    n_problem_type = 'Classification' # 'Regression' # 'Classification'
        #else:
        #    n_problem_type = 'Regression' # 'Regression' # 'Classification'
        
        data_results_coll = {}
        for exp_num in range(EXP_RUN):    
            data_file = data_file_set[j]
            is_norm = True
            isOptimizeParam = False
            normalize = [0.0, 1.0]
            params = SettingParamter.getDataFile(data_file, n_problem_type, is_norm, normalize)
            
            dataProcessing = DataProcessing()
            dataProcessing.setParams(params) 
            data_input_values, data_target_values, random_sequence = data_partition(dataProcessing, params.n_validation_method)
            X_train, X_test, y_train, y_test = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method)
            
            inputs = params.n_max_input_attr
            outputs = params.n_max_target_attr
            
            #TODO: Optimizaer setup
            for solve in solver:
                if(runOpt == '_'):
                    if(solve == 'RMSprop'):
                        opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
                    if(solve == 'Adam'):
                        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
                    if(solve == 'Adagrad'):
                        opt = tf.keras.optimizers.Adagrad(learning_rate=0.1)
                    if(solve == 'SGD'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
                    if(solve == 'MGD'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
                    if(solve == 'NAG'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
                else:
                    if(solve == 'RMSprop'):
                        opt = tf.keras.optimizers.RMSprop() # Defult param
                    if(solve == 'Adam'):
                        opt = tf.keras.optimizers.Adam()
                    if(solve == 'Adagrad'):
                        opt = tf.keras.optimizers.Adagrad()
                    if(solve == 'SGD'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
                    if(solve == 'MGD'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
                    if(solve == 'NAG'):
                        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
                    
                model = models.Sequential()
                
                #TODO: Model configurationj setting
                model.add(Dense(100, activation=FUN, input_shape=(inputs,)))
                #model.add(Dense(100, activation=FUN, input_shape=(inputs,), kernel_regularizer=l1(0.01)))
                #model.add(Dense(100, activation=FUN, input_shape=(inputs,), kernel_regularizer=l2(0.01)))
                #model.add(Dense(100, activation=FUN, input_shape=(inputs,), kernel_regularizer=l1_l2(0.01,0.01)))
                #model.add(Dense(100, activation='sigmoid', input_shape=(inputs,), kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
                
                #TODO: DROPOUT
                #model.add(Dropout(0.4))
    
                #TODO: Call back setting Early Stopping
                es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=ES,restore_best_weights=True)
                
                start_time = time.time()
                if n_problem_type == 'Classification':
                    model.add(Dense(outputs, activation='softmax'))
                    #model.count_params()
                    #model.summary()            
                    # Compile model
                    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                    #es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
        
                    # Train model
                    history = model.fit(X_train, y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              #shuffle=True,
                              callbacks=[es_callback],
                              validation_data=(X_test, y_test),
                              verbose=0)
                    
                    #prediction            
                    #Time computation start
                    start_timeFun = time.time()
                    y_pred = model.evaluate(X_train, y_train, verbose=0)
                    errorTrn = y_pred[1]
                    funcEvltime = (time.time() - start_timeFun) / len(X_train)
                    
                    #Report
                    y_pred = model.evaluate(X_test, y_test, verbose=0)
                    errorTst = y_pred[1]
                    
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(current_time,' ',exp_num, data_file,' ',solve,' test accuracy:', y_pred[1])
                    #collection                
                    error = [errorTrn, errorTst, model.count_params(), funcEvltime, history.history]
                    data_results_coll.update({str(data_file.split('.')[0])+"_"+str(exp_num)+"_"+solve: error})
                else:
                    model.add(Dense(outputs, activation='sigmoid'))
                    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
                    
    
                    # Train model
                    history = model.fit(X_train, y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              #shuffle=True,
                              callbacks=[es_callback],
                              validation_data=(X_test, y_test),
                              verbose=0)
                    
                    start_timeFun = time.time()
                    y_pred = model.evaluate(X_train, y_train, verbose=0)
                    errorTrn = y_pred[1]
                    funcEvltime = (time.time() - start_timeFun) / len(X_train)
                    r2Trn = r2_score(y_train, model.predict(X_train))
                    #Report
                    y_pred = model.evaluate(X_test, y_test, verbose=0)
                    
                    errorTst = y_pred[1]
                    r2Tst = r2_score(y_test, model.predict(X_test))
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(current_time,' ',exp_num, data_file,' ',solve,' test accuracy:', r2Tst)
                    #collection
                    error = [errorTrn, errorTst, r2Trn, r2Tst, model.count_params(), funcEvltime, history.history]
                    data_results_coll.update({str(data_file.split('.')[0])+"_"+str(exp_num)+"_"+solve: error})
                #collect all SGDs
            #end for all solve
        #end for each runs
        data_results_coll.update({'model_config':model.layers[0].get_config()})
        np.save(os.getcwd()+os.sep+'outputs'+os.sep+data_file.split('.')[0]+'_Ep_'+str(EPOCHS)+'_B_'+str(BATCH_SIZE)+'_'+str(FUN)+'_ES_'+str(ES)+'_Reg_'+str(REG)+runOpt, data_results_coll)
    #end for each dataum
#%% save np.load
print('All experiments end. for')
print(model.layers[0].get_config())
print(data_file_set)
#a = np.load('outputs'+os.sep+'mlp_friedman_Ep_500_B1Sig_l2.npy.npy').item()

