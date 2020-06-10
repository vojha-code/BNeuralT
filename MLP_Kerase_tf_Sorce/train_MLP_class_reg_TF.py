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

#data_file_set = ['dee.csv']

#solver=['RMSprop', 'Adam','Adagrad', 'SGD','MGD', 'NAG']

EPOCHS=10 # 500 
BATCH_SIZE = 10 # 1
expleriment_run = 1 # 30

data_results = []
for j in range(len(data_file_set)):
    if j < 9:
        n_problem_type = 'Classification' # 'Regression' # 'Classification'
    else:
        n_problem_type = 'Regression' # 'Regression' # 'Classification'
    n_problem_type = 'Regression' # 'Regression' # 'Classification'
    
    data_results_coll = {}
    for i in range(expleriment_run):    
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
                y_pred = model.evaluate(X_train, y_train, verbose=0)
                errorTrn = y_pred[1]
                #Report
                y_pred = model.evaluate(X_test, y_test, verbose=0)
                print(data_file,' ',solve,' test accuracy:', y_pred[1])
                errorTst = y_pred[1]
                #collection                
                error = [errorTrn, errorTst, model.count_params(), (time.time() - start_time), history.history]
            else:
                model.add(Dense(outputs, activation='sigmoid'))
                model.compile(optimizer=opt, loss='mse', metrics=['mse'])
                history = model.fit(X_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          verbose=0)#,   
                
                y_pred = model.evaluate(X_train, y_train, verbose=0)
                errorTrn = y_pred[1]
                r2Trn = r2_score(y_train, model.predict(X_train))
                #Report
                y_pred = model.evaluate(X_test, y_test, verbose=0)
                
                
                errorTst = y_pred[1]
                r2Tst = r2_score(y_test, model.predict(X_test))
                print(data_file,' ',solve,' test accuracy:', r2Tst)
                #collection
                error = [errorTrn, errorTst, r2Trn, r2Tst, model.count_params(), (time.time() - start_time), history.history]
            #collect all SGDs
            data_results_coll.update({str(data_file.split('.')[0])+"_"+str(i)+"_"+solve: error})
        #end for all solve
    #end for each runs
    np.save('outputs'+os.sep+'keras_'+data_file.split('.')[0]+'_Ep_'+str(EPOCHS)+'_B'+str(BATCH_SIZE)+'D_sig_all', data_results_coll)
#end for each dataum
#%% save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
a = np.load('outputs'+os.sep+'keras_0_Adagrad_Ep_10D_sig_all.npy').item()

