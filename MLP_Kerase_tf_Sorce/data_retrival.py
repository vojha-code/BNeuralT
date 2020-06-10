'''

Anonymous Author(s)

'''
import numpy as np
import argparse
from data_processing import DataProcessing

class SettingParamter():
    '''
        Collecting paramter into arg parse
    '''
    mParser = None
    
    def __init__(self):
        '''
            initiate a parser
        '''
        print('Gathering parameter...')
        self.mParser = argparse.ArgumentParser() # parser accumulats all parametrs 
        
    def setData(self, data_file, data, p_problem_type = 'Classification', normalize = [0.0, 1.0], normlize_data = False):
        '''
            Setting data sets parametre
        '''
        self.mParser.add_argument('--n_dataset_name', default = data_file, type=str)
        self.mParser.add_argument('--n_problem_type', default = p_problem_type, type=str)
        print(' --- Preprocessing loaded data --- ')
        #Creating/ Setting the imput data Matrix        
        # input values data structuer used is -> float64
        self.mParser.add_argument('--n_is_data_normlized', default = normlize_data, type=bool)
        if normlize_data:
            self.mParser.add_argument('--n_data_norm_min', default = normalize[0], type=float)
            self.mParser.add_argument('--n_data_norm_max', default = normalize[1], type=float)
            self.mParser.add_argument('--n_data_input_min', default = data.feature_min, type=float)
            self.mParser.add_argument('--n_data_input_max', default = data.feature_max, type=float)
            if data_file == "mnist.csv":
                data_input_values = data.data
                data_input_values = data_input_values.astype('float32')             
                data_input_values /= 255 # normalizing by deviding by heighst intensity
            else:
                data_input_values = DataProcessing.normalize_data(data.data, normalize[0], normalize[1])
        else:
            data_input_values = data.data               
        data_input_names = data.feature_names # input names data structuer is -> list 
        data_input_attrs = data_input_values.shape[1] # number of input attributes -> int retirve from columns number of input values
        
        
        #setting number of columns (attributes) in the dataset
        self.mParser.add_argument('--n_data_input_values', default = data_input_values, type=float)
        self.mParser.add_argument('--n_data_input_names',  default = data_input_names, type=str)
        self.mParser.add_argument('--n_max_input_attr', default = int(data_input_attrs), type=int)
        
        # preprocessing of the target columns - this should be further  automated 
        #fetching raw data target
        data_target_raw = data.target
        if(p_problem_type == 'Classification'):
            #set the taget names for classification problem
            data_traget_names = data.target_names
            self.mParser.add_argument('--n_data_target_names', default = data_traget_names, type=str)
            
            # Creating taget column
            #Source code:  https://scikit-learn.org/stable/modules/preprocessing_targets.html
            #print('    --- Class lavel binarization ---')
            from sklearn import preprocessing
            le = preprocessing.LabelBinarizer()
            le.fit(data_target_raw)
            #transform single column target to multicolimn target column 
            data_target_values = le.transform(data_target_raw)
            #print(data_target_values)
            if(data_target_values.shape[1] == 1):
                #print('Changed label')
                data_target_values = np.asarray([np.where(data.target == 0, 1, 0), np.where(data.target == 1, 1, 0)]).T

            data_traget_attrs = data_target_values.shape[1]    
            self.mParser.add_argument('--n_data_target_values', default = data_target_values, type=float)
            # set the number or of output columns equivalent to number of classes
            self.mParser.add_argument('--n_max_target_attr',  default = int(data_traget_attrs), type=int)
        else:
            if normlize_data:
                self.mParser.add_argument('--n_data_target_min', default = data.target_min, type=float)
                self.mParser.add_argument('--n_data_target_max', default = data.target_max, type=float)
                data_target_values = DataProcessing.normalize_data(data_target_raw, normalize[0], normalize[1])
            else:
                data_target_values = data_target_raw
                
            self.mParser.add_argument('--n_data_target_values', default = data_target_values, type=float)
            self.mParser.add_argument('--n_data_target_names', default = ['output'], type=str)
            self.mParser.add_argument('--n_max_target_attr', default = 1, type=int)  
        
        self.mParser.add_argument('--n_validation_method', default = 'holdout', type=str, choices = ['holdout','holdout_val','k_fold','five_x_two_fold'])  
        self.mParser.add_argument('--n_validation_folds', default = 10, type=int, choices = [2,5,10,'...'])  

               
    def checkParameters(self, params):
        print('Check data: ',  params.n_dataset_name)
        print('  PARAM: problem type          : ', params.n_problem_type)
        print('  PARAM: exs in instance sapce : ', params.n_data_input_values.shape[0])
        print('  PARAM: max input attributes  : ', params.n_max_input_attr)
        print('  PARAM: max input attributes  : ', params.n_data_input_names)
        print('  PARAM: max output attributes : ', params.n_max_target_attr)
        print('  PARAM: max output attributes : ', params.n_data_target_names)
        print('  PARAM: max output attributes : ', params.n_validation_method)
        print('  PARAM: data normalization    : ', params.n_is_data_normlized)
        if params.n_is_data_normlized:
            print('  PARAM: norm min              : ', params.n_data_norm_min)    
            print('  PARAM: norm max              : ', params.n_data_norm_max)    
            print('  PARAM: data inputs min       : ', params.n_data_input_min)    
            print('  PARAM: data inputs max       : ', params.n_data_input_max)
            if params.n_problem_type != 'Classification':
                print('  PARAM: data target min       : ', params.n_data_target_min)    
                print('  PARAM: data target max       : ', params.n_data_target_max)

    def getDataFile(data_file = 'input_a_file.csv', n_problem_type = 'Classification', is_norm = False, normalize = [0.0, 1.0]):   
        print('Setting problem (task)...') 
        dataProcessing = DataProcessing()
        data = dataProcessing.load_data(n_problem_type, data_file)
        #%
        setParams = SettingParamter()
        setParams.setData(data_file, data, n_problem_type, normalize, is_norm)
        
        #Retriving paramters
        params = setParams.mParser.parse_args()
        setParams.checkParameters(params)#  checking paramters
        return params