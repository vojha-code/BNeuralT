package experimentSettings;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import dataCSVReader.Data;
import dataCSVReader.DataLoader;

public class ExperimentParameters {
	public static String dataPath = System.getProperty("user.dir") + File.separator + "data" + File.separator;

	//private static String exTModelPath = "....";
	//public static String modelPath = exTModelPath + File.separator + "model" + File.separator;
	public static String modelPath = System.getProperty("user.dir") + File.separator + "model" + File.separator;

	public String n_data_name;
	public String n_problem_type = "Classification";
	public boolean n_should_normalize_data = true;
	public double[] n_scale = {0.0, 1.0};

	public String n_validation_method = "holdout";//'holdout','holdout_val','k_fold','five_x_two_fold'
	public double n_training_set_size = 0.8;// out of 100% of all examples
	public double n_validation_set_size = 0.0;// out of 100% of all remaining examples examples after training partition
	public int n_validation_folds = 10;

	public Data dataset;
	public String[] n_data_input_names;
	public int n_max_input_attr;
	public String[] n_data_target_names;
	public int n_max_target_attr;
	public double[] n_data_target_min;
	public double[] n_data_target_max;

	//Neural tree related parameters
	public boolean n_bound_tree_size;
	public int n_min_tree_size_value;
	public int n_max_children = 4;
	public int n_max_depth = 5;
	public double n_probIntLeafNodeGen = 0.4;
	public double[] n_fun_range = { 0.01, 1.0 };//[0.0, 1.0]
	public double[] n_weight_range = { 0.0, 1.0 };
	public String n_fun_type = "sigmoid"; //'Gaussian','tanh', 'sigmoid', 'ReLU', 'softmax'
	public String n_out_fun_type = "sigmoid";//'Gaussian','tanh', 'sigmoid', 'ReLU', 'softmax'

	// Gradient descent related parameters
	public String n_param_optimizer = "gd"; //['gd','mh']
	public int n_param_opt_max_itr = 10;
	public String n_algo_param = "rmsprop"; //['gd','momentum_gd','nesterov_accelerated_gd','adagrad','rmsprop','adam'])
	public String n_gd_eval_mode = "stochastic"; //['batch','stochastic'])
	public int n_batch_size = 10;
	public double n_gd_precision = 0.00000001;//'Termination tolerance')
	public double n_gd_eta = 0.1; //'gradient descent learning rate')
	public double n_gd_gamma = 0.9; //'gradient descent momentum rate')
	public double n_gd_eps = 1e-8; //'gradient epsilon')
	public double n_gd_beta = 0.9; //'gradient beta')
	public double n_gd_beta1 = 0.9; //'gradient beta')
	public double n_gd_beta2 = 0.9;//'gradient beta')
	public String n_check_epoch_set = "test";




	public ExperimentParameters(String data_name, String problem_type, boolean should_normalize_data, double[] scale, String param_opt, File directory, String trial) {
		this.n_data_name = data_name;
		this.n_problem_type = problem_type;
		this.n_should_normalize_data = should_normalize_data;
		this.n_scale = scale;
		//'holdout','holdout_val','k_fold','five_x_two_fold'
		this.n_validation_method = "holdout";
		this.n_validation_folds = 2; // only for k_fold
		try {
			setParameters(param_opt);
			//saveExperinetParameter(directory, trial);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} // defaults parameters are set
	}

	public ExperimentParameters(String data_name, LinkedHashMap<String, Object> exp_file_setup) {
		this.n_data_name = data_name; //(String) exp_file_setup.get("n_data_name");
		this.n_problem_type = (String) exp_file_setup.get("n_problem_type");
		this.n_should_normalize_data = (boolean) exp_file_setup.get("n_should_normalize_data");
		this.n_scale = (double[]) exp_file_setup.get("n_scale");
		this.n_validation_method = (String) exp_file_setup.get("n_validation_method");
		this.n_validation_folds = (int) exp_file_setup.get("n_validation_folds"); // only for k_fold
		try {
			setDataValues();
		} catch (IOException e) {
			e.printStackTrace();
		}

		this.n_bound_tree_size = (boolean) exp_file_setup.get("n_bound_tree_size");
		this.n_min_tree_size_value = (int) exp_file_setup.get("n_min_tree_size_value");

		this.n_max_children = (int) exp_file_setup.get("n_max_children");
		this.n_max_depth = (int) exp_file_setup.get("n_max_depth");
		this.n_probIntLeafNodeGen = (double) exp_file_setup.get("n_prob_of_int_leaf_gen");

		this.n_fun_range = (double []) exp_file_setup.get("n_fun_range");;
		this.n_weight_range = (double []) exp_file_setup.get("n_weight_range");
		this.n_fun_type = (String) exp_file_setup.get("n_fun_type");
		this.n_out_fun_type = (String) exp_file_setup.get("n_out_fun_type");


		this.n_param_optimizer =  (String) exp_file_setup.get("n_param_optimizer");
		this.n_param_opt_max_itr = (int) exp_file_setup.get("n_param_opt_max_itr");
		this.n_algo_param = (String) exp_file_setup.get("n_algo_param");

		this.n_gd_eval_mode = (String) exp_file_setup.get("n_gd_eval_mode");
		this.n_batch_size = (int) exp_file_setup.get("n_gd_batch_size");
		this.n_gd_precision = (double) exp_file_setup.get("n_gd_precision");

		this.n_gd_eta = (double) exp_file_setup.get("n_gd_eta");
		this.n_gd_gamma = (double) exp_file_setup.get("n_gd_gamma");
		this.n_gd_eps = (double) exp_file_setup.get("n_gd_eps");
		this.n_gd_beta = (double) exp_file_setup.get("n_gd_beta");
		this.n_gd_beta1 = (double) exp_file_setup.get("n_gd_beta1");
		this.n_gd_beta2 = (double) exp_file_setup.get("n_gd_beta2");
		this.n_check_epoch_set  = (String) exp_file_setup.get("n_check_epoch_set");

		checkParamters();
	}



	private void checkParamters() {

		System.out.println("Data and data pre-prpciessing description");
		System.out.println("	Data Name             : "+n_data_name);
		System.out.println("	Problem Tyep          : "+n_problem_type);
		System.out.println("	Data Normalization    : "+n_should_normalize_data);
		System.out.println("	Normlization Scale    : "+Arrays.toString(n_scale));
		System.out.println("	Validation Method     : "+n_validation_method);
		if(n_validation_method.equalsIgnoreCase("holdout")) {
			System.out.println("	Training Set Size     : "+n_training_set_size);
		}
		//System.out.println(" Validation Set        :"+n_validation_set_size);
		//System.out.println(" Validation Fold       :"+n_validation_folds);


		System.out.println("BNerualT propoerties");
		System.out.println("	Bound Tree Size       : "+n_bound_tree_size);
		if(n_bound_tree_size) {
			System.out.println("	Min Tree Size         : "+n_min_tree_size_value);
		}
		System.out.println("	Max Child Per Node    : "+n_max_children);
		System.out.println("	Max Tree Depth        : "+n_max_depth);
		System.out.println("	Prob. Leaf Internal   : "+n_probIntLeafNodeGen);


		System.out.println("	Bound Tree Size       : "+n_fun_type);
		System.out.println("	Bound Tree Size       : "+n_out_fun_type);
		if(n_fun_type.equalsIgnoreCase("Gaussian")) {
			System.out.println(" Function Range        : "+Arrays.toString(n_fun_range));
		}
		System.out.println("	Edage weight bound    : "+Arrays.toString(n_weight_range));

		System.out.println("GD Optimizers Paramters");
		System.out.println("	Optimizer mode        : "+n_param_optimizer);
		System.out.println("	Max Epochs            : "+n_param_opt_max_itr);
		System.out.println("	Optimizer (Algo)      : "+n_algo_param);
		System.out.println("	Learning mode         : "+n_gd_eval_mode);
		System.out.println("	Batch size            : "+n_batch_size);
		System.out.println("	Precision             : "+n_gd_precision);
		System.out.println("	Learning rate         : "+n_gd_eta);
		System.out.println("	Momentum rate         : "+n_gd_gamma);
		System.out.println("	Epsilon               : "+n_gd_eps);
		System.out.println("	Beta                  : "+n_gd_beta);
		System.out.println("	Beta1 (Adam)          : "+n_gd_beta1);
		System.out.println("	Beta2 (Adam)          : "+n_gd_beta2);
	}

	private void setDataValues() throws IOException {
		DataLoader dl = new DataLoader();
		this.dataset = dl.readCSVFile(n_data_name, n_problem_type, n_should_normalize_data, n_scale);

		this.n_data_input_names = dataset.feature_names;
		this.n_max_input_attr = dataset.feature_names.length;

		this.n_data_target_names = dataset.target_names;
		this.n_max_target_attr = dataset.target_names.length;

		if (!n_problem_type.equalsIgnoreCase("Classification")) {
			this.n_data_target_min = dataset.target_min;
			this.n_data_target_max = dataset.target_max;
		}
	}



	private void setTreeParameters() {
		//TODO:  Change these parameters
		this.n_max_children = 5; // hyper parameters <-------------------------------------
		this.n_max_depth = 5; // <---------------------------------------------------------
		this.n_probIntLeafNodeGen = 0.5;// probability of internal node being a leaf node

		double[] fun_range = { 0.01, 1.0 };
		double[] weight_range = { 0.0, 1.0 };
		this.n_fun_range = fun_range;
		this.n_weight_range = weight_range;
		this.n_fun_type = "sigmoid"; 		//'Gaussian','tanh', 'sigmoid', 'ReLU',
		this.n_out_fun_type = "sigmoid"; //  'softmax' found not fit for BPNT

	}

	private void setParamterOptimizationGD() {
		//['gd','momentum_gd','nesterov_accelerated_gd','adagrad','rmsprop','adam'])
		this.n_param_opt_max_itr = 5; //TODO
		this.n_algo_param = "nesterov_accelerated_gd";
		this.n_gd_eval_mode = "stochastic"; //['batch','stochastic','mini_batch'])
		this.n_batch_size = 128; // only for mini-batch
		this.n_gd_precision = 0.00000001; ////'Termination tolarane')
		this.n_gd_eta = 0.001; //'gradient descent learning rate')
		this.n_gd_gamma = 0.9; //'gradient descent momentum rate')
		this.n_gd_eps = 1e-8; //'gradient epsilon')
		this.n_gd_beta = 0.9; //'gradient beta')
		this.n_gd_beta1 = 0.9; //'gradient beta')
		this.n_gd_beta2 = 0.9; //'gradient beta')
	}

	public void setParameters(String param_opt) throws IOException {
		setDataValues();
		setTreeParameters();
		this.n_param_optimizer = param_opt; //['gd','mh']
		setParamterOptimizationGD();
	}

	public void saveExperinetParameter(File directory, String trial, String paramAlgoName, ArrayList<Object> listSavePerfroance) {
		LinkedHashMap<String, Object> hashmap = new LinkedHashMap<String, Object>();
		hashmap.put("n_data_name", n_data_name);
		hashmap.put("n_problem_type", n_problem_type);
		hashmap.put("n_should_normalize_data", n_should_normalize_data);
		hashmap.put("n_scale", n_scale);

		hashmap.put("n_data_input_names", n_data_input_names);
		hashmap.put("n_max_input_attr", n_max_input_attr);
		hashmap.put("n_data_target_names", n_data_target_names);
		hashmap.put("n_max_target_attr", n_max_target_attr);
		hashmap.put("n_data_target_min", n_data_target_min);
		hashmap.put("n_data_target_max", n_data_target_max);

		hashmap.put("n_validation_method", n_validation_method);//'holdout','holdout_val','k_fold','five_x_two_fold'
		hashmap.put("n_training_set_size", n_training_set_size);// out of 100% of all examples
		hashmap.put("n_validation_set_size", n_validation_set_size);// out of 100% of all remaining examples examples after training partition
		hashmap.put("n_validation_folds", n_validation_folds);

		hashmap.put("n_max_children", n_max_children);
		hashmap.put("n_max_depth", n_max_depth);
		hashmap.put("n_prob_of_int_leaf_gen", n_probIntLeafNodeGen);
		hashmap.put("n_fun_range", n_fun_range);//[0.0, 1.0]
		hashmap.put("n_weight_range", n_weight_range); //[0.0, 1.0]
		hashmap.put("n_fun_type", n_fun_type);
		hashmap.put("n_out_fun_type", n_out_fun_type);


		hashmap.put("n_param_optimizer", n_param_optimizer); //['gd','mh']
		hashmap.put("n_param_opt_max_itr", n_param_opt_max_itr);
		hashmap.put("n_algo_param", paramAlgoName);

		if (n_param_optimizer.equalsIgnoreCase("gd")) {
			hashmap.put("n_gd_eval_mode", n_gd_eval_mode); //['batch','stochastic'])
			hashmap.put("n_gd_batch_size", n_batch_size);//'Termination tolerance')
			hashmap.put("n_gd_precision", n_gd_precision);//'Termination tolerance')
			hashmap.put("n_gd_eta", n_gd_eta); //'gradient descent learning rate')
			hashmap.put("n_gd_gamma", n_gd_gamma); //'gradient descent momentum rate')
			hashmap.put("n_gd_eps", n_gd_eps); //'gradient epsilon')
			hashmap.put("n_gd_beta", n_gd_beta); //'gradient beta')
			hashmap.put("n_gd_beta1", n_gd_beta1); //'gradient beta')
			hashmap.put("n_gd_beta2", n_gd_beta2); //'gradient beta')
		} else {
			//TODO entries for mh will  come here...
		}

		ArrayList<Object> nBestCostsTrain = (ArrayList<Object>) listSavePerfroance.get(0); // Save Best Structure Training Error
		ArrayList<Object> nBestCostsTest = (ArrayList<Object>) listSavePerfroance.get(1);// Save Best Structure Test Error
		if (n_problem_type.equalsIgnoreCase("Classification")) {
			hashmap = putClassification(hashmap, nBestCostsTrain, "train");
			hashmap = putClassification(hashmap, nBestCostsTest, "test");
			hashmap.put("n_tree_size", nBestCostsTrain.get(8)); //tree size
		} else {
			hashmap = putRegression(hashmap, nBestCostsTrain, "train");
			hashmap = putRegression(hashmap, nBestCostsTest, "test");
			hashmap.put("n_tree_size", nBestCostsTrain.get(3)); //tree size
		}


		hashmap.put("n_data_rand_secquence", listSavePerfroance.get(2)); // Save Data Random Sequence
		hashmap.put("n_train_time_sec", (double) listSavePerfroance.get(3)); // Save time elapsed

		/*JSONObject json = new JSONObject();
		json.putAll(hashmap);
		FileWriter fwExpParams;
		try {
			fwExpParams = new FileWriter(directory + File.separator + "experiment_" + trial + ".json");
			fwExpParams.write(json.toJSONString());
			fwExpParams.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} */
		//System.out.print(jsonStr);

		File file = new File(directory + File.separator + "experiment_" + trial + ".txt");
		BufferedWriter bf = null;
		try {
			//create new BufferedWriter for the output file
			bf = new BufferedWriter(new FileWriter(file));
			int lastKey = hashmap.size();
			int index = 0;
			//iterate map entries
			bf.write("{"); // add a braces (right)
			for (Map.Entry<String, Object> entry : hashmap.entrySet()) {
				String key = entry.getKey();
				Object value = entry.getValue();
				if (value != null) {
					String valueStr = value.toString();
					//put key and value separated by a colon
					if (isPrimitiveArray(value)) { // if(value.getClass().isArray()) { //
						if (value instanceof String[]) {
							valueStr = Arrays.toString((String[]) value);
						}
						if (value instanceof Integer[]) {
							valueStr = Arrays.toString((Integer[]) value);
						}
						if (value instanceof double[]) {
							valueStr = Arrays.toString((double[]) value);
						}
						if (value instanceof int[]) {
							valueStr = Arrays.toString((int[]) value);
						}
					}
					bf.write("\"" + key + "\": \"" + valueStr + "\"");
					if (index != lastKey - 1)
					{
						bf.write(","); // add a comma
					}
					//new line
					bf.newLine();
				} //if null do not include in the parameter setting
				index++;
			}
			bf.write("}"); // add a braces ()
			bf.flush();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				//always close the writer
				bf.close();
			} catch (Exception e) {
			}
		}
	}// end parameter saving

	public static boolean isPrimitiveArray(Object obj) {
		//return obj != null && obj.getClass().isArray() && obj.getClass().getComponentType() != null && obj.getClass().getComponentType().isPrimitive();
		return obj != null && obj.getClass().isArray();
	}

	private LinkedHashMap<String, Object> putClassification(LinkedHashMap<String, Object> hashmap, ArrayList<Object> nBestCosts, String pSet) {
		hashmap.put("n_" + pSet + "_error", nBestCosts.get(0)); // error (double)
		hashmap.put("n_" + pSet + "_prec", nBestCosts.get(1));//PPR (double[])
		hashmap.put("n_" + pSet + "_recall", nBestCosts.get(2));//recall (double[])
		hashmap.put("n_" + pSet + "_spec", nBestCosts.get(3));//TNR (double[])

		hashmap.put("n_" + pSet + "_tp", nBestCosts.get(4));//tp (int[])
		hashmap.put("n_" + pSet + "_fp", nBestCosts.get(5));//fp (int[])
		hashmap.put("n_" + pSet + "_fn", nBestCosts.get(6));//fn (int[])
		hashmap.put("n_" + pSet + "_tn", nBestCosts.get(7)); //tn (int[])
		return hashmap;
	}

	private LinkedHashMap<String, Object> putRegression(LinkedHashMap<String, Object> hashmap, ArrayList<Object> nBestCosts, String pSet) {
		//System.out.println(pSet+":"+nBestCosts);
		hashmap.put("n_" + pSet + "_error", nBestCosts.get(0)); //mse (double)
		hashmap.put("n_" + pSet + "_corr", nBestCosts.get(1));// corr (double)
		hashmap.put("n_" + pSet + "_r2", nBestCosts.get(2));// r2 (double)
		return hashmap;
	}
}//end class
