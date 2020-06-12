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
#%%Clasification
#DATA PREPARATION
data_file_set = [
                'iris',
                'wisconsin',
                'wine',
                'heart.csv',
                'ionosphere.csv',
                'australian.csv',
                'pima.csv',
                'glass.csv',
                'vehicle.csv',
                'dee.csv',
                'diabetese.csv',
                'baseball.csv',
                'friedman.csv',
                'mpg6.csv']

#data_file_set = ['iris']

solver=['RMSprop', 'Adam','Adagrad', 'SGD','MGD', 'NAG']

EPOCHS=10 # 500 
BATCH_SIZE = 50 # 1
expleriment_run = 10 # 30

data_time_results = {}
for j in range(len(data_file_set)):
    if j < 9:
        n_problem_type = 'Classification' # 'Regression' # 'Classification'
    else:
        n_problem_type = 'Regression' # 'Regression' # 'Classification'
    #n_problem_type = 'Regression' # 'Regression' # 'Classification'

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
    
    print(data_file)    
    for solve in solver:
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
        
        model = models.Sequential()
        model.add(Dense(100, activation='sigmoid', input_shape=(inputs,)))
        
        start_time = time.time()
        if n_problem_type == 'Classification':
            model.add(Dense(outputs, activation='softmax'))
            #model.count_params()
            #model.summary()            
            # Compile model
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            # Train model
            history = model.fit(X_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=0)#,
                      #validation_data=(X_test, y_test))
            #prediction            
            timeColl = []
            for i in range(expleriment_run):
                #Time computation start
                start_timeFun = time.time()
                y_pred = model.evaluate(X_test, y_test, verbose=0)
                errorTrn = y_pred[1]
                #Time computation end
                funcEvltime = (time.time() - start_timeFun)
                timeColl.append(funcEvltime)
            print('    ',solve,' time tau:', np.mean(timeColl))
            data_time_results.update({str(data_file.split('.')[0])+"_"+solve: np.mean(timeColl)})
        else:
            model.add(Dense(outputs, activation='sigmoid'))
            model.compile(optimizer=opt, loss='mse', metrics=['mse'])
            history = model.fit(X_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=0)#,   
            
            timeColl = []
            for i in range(expleriment_run):
                #Time computation start
                start_timeFun = time.time()
                y_pred = model.evaluate(X_test, y_test, verbose=0)
                errorTrn = y_pred[1]
                r2Trn = r2_score(y_train, model.predict(X_train))
                 #Time computation end
                funcEvltime = (time.time() - start_timeFun)
                timeColl.append(funcEvltime)
            print('    ',solve,' time tau:', np.mean(timeColl))
            data_time_results.update({str(data_file.split('.')[0])+"_"+solve: np.mean(timeColl)})

print("\n\nForward pass time in seconds")            
for keys in data_time_results:
    print(keys," : ", data_time_results[keys])
