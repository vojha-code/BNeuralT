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

## Hyperparamter setting

json format hyperparamter experiment setup files

```diff
{
"n_num_exp": "1",     - number of times an experment to be rapeated
"n_data_name": "iris.csv", 
"n_problem_type": "Classification",
"n_should_normalize_data": "true", 
"n_scale": "[0.0, 1.0]", 
"n_validation_method": "holdout", 
"n_training_set_size": "0.8", <br>
"n_validation_set_size": "0.0", <br>
"n_validation_folds": "2", <br>
"n_bound_tree_size": "false", <br>
"n_min_tree_size_value": "100", <br>
"n_max_children": "4", <br>
"n_max_depth": "4", <br>
"n_prob_of_int_leaf_gen": "0.6", <br>
"n_fun_range": "[0.01, 1.0]", <br>
"n_weight_range": "[0.0, 1.0]", <br>
"n_fun_type": "sigmoid", <br>
"n_out_fun_type": "sigmoid", <br>
"n_param_optimizer": "gd", <br>
"n_algo_param": "rmsprop", <br>
"n_gd_eval_mode": "stochastic", <br>
"n_gd_batch_size": "10" <br>
"n_gd_precision": "0.0000000001", <br>
"n_gd_eta": "0.01",<br>
"n_gd_gamma": "0.9", <br>
"n_gd_eps": "0.0000000001", <br>
"n_gd_beta": "0.9", <br>
"n_gd_beta1": "0.9", <br>
"n_gd_beta2": "0.9", <br>
"n_param_opt_max_itr": "10", <br>
"n_check_epoch_set": "test" <br>
}
```

## Model Evaluation
...



## Pre-trained models
..
