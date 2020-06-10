# Backpropagation Neural Tree
Ad-hoc Neural Tree Generation and training using Backpropagation Algorithm


## Algorithm
BNeuralT is a machine learning algorithm for learning from data. BNeuralT is applied three categories of learning problems in the examples given here: classification, regression and pattern recognition. The strngth oif algorithm is in its effeciency and robustness in producing low compplexity and high perfroaing models. 


## Dependencies and configurations

The BNeuralT algorithm is written in [Java version 8](https://java.com/en/download/) in Eclipse version 2020‑03 and has the following dependencises. 
  -The algorithm uses DenseMatrix64 of [EJML](http://ejml.org/wiki/index.php?title=Main_Page)
  - JSON object [json-simple-1.1](https://mvnrepository.com/artifact/com.googlecode.json-simple/json-simple/1.1.1) for saving trained models in json format. 

The MLP algorithm and python scripts is written in [Python 3.5](https://www.python.org/downloads/release/python-350/) and has the following dependencises. 
  - [tensorflow 2.1](https://www.tensorflow.org/install)
  - [Keras 2.3](https://keras.io/)
  - [NumPy 1.17.5](https://numpy.org/)
  - [scikit-learn 0.19.1](https://scikit-learn.org/stable/)
  - [scipy 1.4.1](https://www.scipy.org/)
  - [pandas 0.23](https://pandas.pydata.org/)


## Project structure, data, and source code files
###### project directory structure 
Setup of exlipse project strcutre is as follows:
- BNeuralT (root)
  - bin
  - data (*csv data files*)
  - dependencies (EJML and JSON)
  - model
    - view (*javascript and HTML files for tree models*)
  - src (Java version 8 source files)
  - trained_models 
###### datasets
All csv files for classification and regression learning problems are in directory *data*

MNIST csv files is too big to upload on GitHub - can be downloaded and put in the folder *data*
MNIST files can be downloaded from
  - [mnist.csv](https://drive.google.com/file/d/1MVmbNkIsYvxS3apGiyOrAq_dpC5OQ8X_/view) processed to fit with data pre-processing
  - [The MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

###### project directory structure 

## Model Evaluation
### Step 1 - test accuracry collection from pre-trained models
Runable JAR files that can be directly run from command line

```diff
# evaluation of classification and regression problems pre-trained models test accuracy collection
- java -Xms7000m -Xmx7000m -jar evaluateTreeModels.jar

# evaluation of pattern recognition (MNIST) pre-trained models and reproducing results in Table 2
- java -Xms7000m -Xmx7000m -jar evaluateTreeModels.jar mnist
```

Eclipse project flise structure has a folder **src** in the folder under the package **trainAndEvaluateTree** main entry point for the models evaluation is:
```diff
+ EvaluateTreeModels.java
```
Under eclipse project necessary *run configuration* setup is as follows:
* Command argument option: 
  - < empty >
  - < mnist >
* VM-Argument option: <-Xms7000m -Xmx7000m>

### Step 2 - running python scripts for *results* CSV files 
Python 3.5 and above version will run the following scripts

```diff
# evaluation of classification and regression problems pre-trained models and reproducing results in Table 1
! python evaluateTreeModels.py

# evaluation of classification and regression problems mean and standard deviation Table
! python evaluateTreeModelsMeanStd.py

# evaluation of classification and regression problems Welch's t-test Table
! python evaluateTreeModelsStats.py
```

### Pre-trained models files and folder structure setup for model evaluation

- BNeuralT
  - trained_models
    - pre_trained_mnist_models_tab2 (MNIST dataset models)
      - mnist_pre_trained_models
        - BNeuralT 10K
        - BNeuralT 18K
        - BNeuralT 20K
        - BNeuralT 200K
    - pre_trained_class_reg_models
      - dir of a dataset 
        - 30 dir <data_name instance number>
          - 6 pre trained model files <Optimizer name models [DOT] json> 
          - 1 experiment files <"experiment" [DOT] txt> preserverd training data sequence
  - BNeuralT_pre_trained_class_reg_models_coll.csv
  - MLP_Keras_TF_models_coll.csv
  - Table_1_class_reg_BNeuralT_MLP_models.csv
  - Table_1_mean_std_class_reg_BNeuralT_MLP_models.csv
  - Table_1_stats_class_reg_BNeuralT_MLP_models.csv
  - Table_2_BNeuralT_models.csv



## Model Training 

### Model training using command line 
Runable JAR files that can be directly run from command line

```diff
# running cost fucntions (tree) for each example in Java Threads (parallel loop) -  effective for large problems only  
- java -Xms7000m -Xmx7000m -jar trainTreeModel_Prll.jar

# running cost fucntions (tree) for each example sequentially (native loop) -  effective for small problems only
- java -Xms7000m -Xmx7000m -jar trainTreeModel_Seq.jar
```

### Model training using source code [Eclipse project]
Eclipse project flise structure has a folder **src** in the folder under the package **trainAndEvaluateTree** main entry point of the training models is:
```diff
+ TrainTree.java
```

Under eclipse project necessary *run configuration* setup is as follows:
* Command argument option: < empty >
* VM-Argument option: <-Xms7000m -Xmx7000m>


### Hyperparamter setting
experiment_training_setup.txt is a json format hyperparamter experiment setup arrangement of hyperparamters.

```diff
{"n_num_exp": "1",                  [1,2,3... ] - number of times an experiment to be repeated
"n_data_name": "mnist.csv",  -      [<data name><dot>csv ] - name of a dataset
"n_problem_type": "Classification", ["Classification", "regression"] - learning problem type for data pre-processing module depends on problem type definition  
"n_should_normalize_data": "true",  ["true", "false"]  - for regression problems normalization is efective for gradient descent
"n_scale": "[0.0, 1.0]",            scaling factor -  [0.0, 1.0]  is best suited. 
"n_validation_method": "holdout",   ["holdout", "k_fold", "five_x_two_fold"]
"n_training_set_size": "0.8",       - a value in [0.0 - 1.0] its training set size, 0.8 indicates 80%, the rest is test set 
"n_validation_folds": "2",          - if "_validation_method" is set to "k_fold"
"n_bound_tree_size": "false",       ["true" , "false"] -  for regression setting it to true is effective 
"n_min_tree_size_value": "7",       [3,4,5,...] setting it to 7 for regression is effective
"n_max_children": "4",              [2,3,4....] maximum child a node can take
"n_max_depth": "4",                 [1,2,3,...] maximum tree depth - increasing it in small amount icreamently is effective
"n_prob_of_int_leaf_gen": "0.6",    - a value in [0.0 - 1.0] its probability of an internal node is a leaf (terminal) node - effective for reducing tree size
"n_weight_range": "[0.0, 1.0]",     - neural weight initialization
"n_fun_type": "sigmoid",            ["sigmoid", "tanh"]- current implementation take sigmoid for "tanh" and other function enable (uncomment) the implementation or implement them
"n_out_fun_type": "sigmoid",        ["sigmoid", "tanh"]- current implementation take sigmoid for "tanh" and other function enable (uncomment) the implementation or implement them
"n_algo_param": "rmsprop",          ["gd","momentum_gd","nesterov_accelerated_gd","adagrad","rmsprop","adam"] - gradient descent optimizers
"n_gd_eval_mode": "stochastic",     ["stochastic", "mini_batch", "batch"] - stochastic and mini_batch are efective
"n_gd_batch_size": "10"             [1,2,3....] a number appropriate (smaller than training set size)
"n_gd_precision": "0.0000000001",   - precision of weight update check
"n_gd_eta": "0.1",                  - a value in [0.0 - 1.0] learning rate [0.1,0.01, 0.001] are effective learning rates in decreasing order of learning speed
"n_gd_gamma": "0.9",                - a value in [0.0 - 1.0] momentum rate
"n_gd_beta": "0.9",                 - a value in [0.0 - 1.0] decay rates (RMSprop)
"n_gd_beta1": "0.9",                - a value in [0.0 - 1.0] decay rates (Adam) 
"n_gd_beta2": "0.9",                - a value in [0.0 - 1.0] decay rates (Adam)
"n_param_opt_max_itr": "10",        [1,2,3,...] gradient descent learning epochs -  balance it with learning rate 
"n_check_epoch_set": "test"}        ["train", "test"]  check models performance on training set and test set during the learning it on training set.
```

