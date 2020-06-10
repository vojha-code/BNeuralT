# Backpropagation Neural Tree
Ad-hoc Neural Tree Generation and training using Backpropagation Algorithm


## Algorithm
...


## Dependencies and configurations
...


## Project structure, data, and source code files
...



## Model Training 

### Model training using command line 
Runable JAR files that can be directly run from command line

```diff
- java -Xms7000m -Xmx7000m -jar trainTreeModel_Prll.jar
- java -Xms7000m -Xmx7000m -jar trainTreeModel_Seq.jar
```

### Model training using source code [Eclipse project]
Eclipse project flise structure has a folder **src** in the folder under the package **trainAndEvaluateTree** main entry point of the training models is:
```diff
+ TrainTree.java
```

Under eclipse project *run configuration* setup is as follows
Command argument option: <empty>
VM-Argument option: <-Xms7000m -Xmx7000m>


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

## Model Evaluation
...



## Pre-trained models
..
