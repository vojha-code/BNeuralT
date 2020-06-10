# Backpropagation Neural Tree
Ad-hoc Neural Tree Generation and training using Backpropagation Algorithm


## Algorithm
...


## Dependencies and configurations
...


## Project structure, data, and source code files
...



## Model Training 
...
![formula](https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1)
### Hyperparamter setting
experiment_training_setup.txt is a json format hyperparamter experiment setup arrangement of hyperparamters.

```diff
{"n_num_exp": "1",                  [options: 1,2,3... ] - number of times an experment to be rapeated
"n_data_name": "mnist.csv",  -      [options: <data name><dot>csv ] name of a dataset
"n_problem_type": "Classification", [options: "Classification", "regression"] - leanring problem type for data pre-processing module depends on problem type defination  
"n_should_normalize_data": "true",  [options: "true", "false"]
"n_scale": "[0.0, 1.0]",            [options: 
"n_validation_method": "holdout",   [options: 
"n_training_set_size": "0.8",       [options:   
"n_validation_set_size": "0.0",     [options: 
"n_validation_folds": "2",          [options: 
"n_bound_tree_size": "false",       [options: 
"n_min_tree_size_value": "100",     [options:  
"n_max_children": "4",              [options: 
"n_max_depth": "4",                 [options: 
"n_prob_of_int_leaf_gen": "0.6",    [options: 
"n_fun_range": "[0.01, 1.0]",       [options: 
"n_weight_range": "[0.0, 1.0]",     [options: 
"n_fun_type": "sigmoid",            [options: 
"n_out_fun_type": "sigmoid",        [options:  
"n_param_optimizer": "gd",          [options: 
"n_algo_param": "rmsprop",          [options: 
"n_gd_eval_mode": "stochastic",     [options: 
"n_gd_batch_size": "10"             [options: 
"n_gd_precision": "0.0000000001",   [options:  
"n_gd_eta": "0.01",                 [options: 
"n_gd_gamma": "0.9",                [options: 
"n_gd_eps": "0.0000000001",         [options: 
"n_gd_beta": "0.9",                 [options: 
"n_gd_beta1": "0.9",                [options: 
"n_gd_beta2": "0.9",                [options: 
"n_param_opt_max_itr": "10",        [options: 
"n_check_epoch_set": "test"}        [options:  
```

## Model Evaluation
...



## Pre-trained models
..
