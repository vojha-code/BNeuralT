/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trainAndEvaluateTree;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedHashMap;

import dataCSVReader.DataPartitioner;
import experimentSettings.ExperimentParameters;
import experimentSettings.ReadExperiment;
import neuralTree.EvaluateTree;
import neuralTree.NeuralTree;
import optimization.Individual;
import optimization.ParamterOptimization;

/**
 *
 * @author yl918888
 */
public class TrainTree {

	@SuppressWarnings("unused")
	private void parameterOptimization(ExperimentParameters exp_params, NeuralTree n_tree, Integer[] data_seqence, File directory, String trial) {
		long start = System.currentTimeMillis();
		//System.out.println("Generating a Random Tree:");

		DataPartitioner cv = new DataPartitioner(data_seqence);
		HashMap<String, Object> dataMap = cv.holdout_method(exp_params.dataset, 0.8, 0.0);
		double[][] train_In = (double[][]) dataMap.get("train_in");
		double[][] train_Tr = (double[][]) dataMap.get("train_tr");
		//Test
		double[][] test_In = (double[][]) dataMap.get("test_in");
		double[][] test_Tr = (double[][]) dataMap.get("test_tr");
		//Val 	//double[][] val_In = (double[][]) dataMap.get("val_in"); //double[][] val_Tr = (double[][]) dataMap.get("val_tr");
		//Integer[] rand_seq = (Integer[])  	dataMap.get("rand_seq");	//Rand seq

		ArrayList<String> expAlgo = new ArrayList<String>();
		if(exp_params.n_algo_param.equalsIgnoreCase("all")) {
			expAlgo.add("rmsprop");
			expAlgo.add("adam");
			expAlgo.add("momentum_gd");
			expAlgo.add("nesterov_accelerated_gd");
			expAlgo.add("adagrad");
			expAlgo.add("gd");
		}else {
			expAlgo.add(exp_params.n_algo_param);
		}

		expAlgo.parallelStream().forEach(algoName -> {
			EvaluateTree ev = new EvaluateTree(train_In, train_Tr, test_In, test_Tr, null, null, exp_params.n_data_target_min, exp_params.n_data_target_max, exp_params.n_scale, exp_params.n_problem_type);
			ParamterOptimization paramters = new ParamterOptimization(ev, exp_params, n_tree.copy_Tree());
			Individual nBestTreeIndividual = paramters.optimize(directory, trial, algoName);
			long finish = System.currentTimeMillis();
			double timeElapsed = (finish - start) / 1000.0;
			System.out.println("Data data: " + exp_params.n_data_name);
			System.out.println("Training time: " + timeElapsed + " sec.");

			ArrayList<Object> perfrormanceSetTest = null;
			ArrayList<Object> perfrormanceSetTrain = null;
			if(exp_params.n_check_epoch_set.equalsIgnoreCase("test")) {
				ev.set_dataset_to_evaluate("train");
				ev.getTreePredictedOutputs(nBestTreeIndividual.mTree);
				perfrormanceSetTrain = ev.getTreeFitness();
				perfrormanceSetTrain.add(nBestTreeIndividual.mCost[1]);

				perfrormanceSetTest = nBestTreeIndividual.mCostAll;
				perfrormanceSetTest.remove(perfrormanceSetTest.size()-1);

				System.out.println("Train Error: " + perfrormanceSetTrain.get(0));
				System.out.println("Test Error: " + nBestTreeIndividual.mCost[0] + " Size: " + nBestTreeIndividual.mCost[1]);// + "_" + nBestTreeIndividual.mCostAll);
			}else {
				ev.set_dataset_to_evaluate("test");
				ev.getTreePredictedOutputs(nBestTreeIndividual.mTree);
				perfrormanceSetTest = ev.getTreeFitness();
				perfrormanceSetTrain = nBestTreeIndividual.mCostAll;
				System.out.println("Train Error: " + nBestTreeIndividual.mCost[0] + " Size: " + nBestTreeIndividual.mCost[1]);// + "_" + nBestTreeIndividual.mCostAll);
				System.out.println("Test Error: " + perfrormanceSetTest.get(0));
			}
			//Save Tree Model of the Best Structure
			nBestTreeIndividual.mTree.saveTreeModel(nBestTreeIndividual.mTree, directory, (trial + "_optPRM_" + algoName), exp_params.n_max_target_attr, exp_params.n_data_input_names, exp_params.n_data_target_names, false);
			ArrayList<Object> listSavePerfroance = new ArrayList<Object>();
			listSavePerfroance.add(perfrormanceSetTrain); // Save Best Structure Training Error
			listSavePerfroance.add(perfrormanceSetTest);// Save Best Structure Test Error
			listSavePerfroance.add(dataMap.get("rand_seq")); // Save Data Random Sequence
			listSavePerfroance.add(timeElapsed); // Save time elapsed
			exp_params.saveExperinetParameter(directory, trial + "_"+algoName, algoName, listSavePerfroance);
		});
	}//end parameter optimisation


	private static void runExperimentNew(Integer itr, String data_name, LinkedHashMap<String, Object> exp_file_setup) {
		System.out.println("Tree Experiments Starts..." + itr);
		// Chose a data-set for tree experiment
		String file_name = data_name.split("\\.")[0];
		String trial = file_name + "_" + itr + "_" + (new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(Calendar.getInstance().getTime()));
		//trial = file_name + "_" + itr + "_" + trial;
		//trial = "";
		File directory = new File(ExperimentParameters.modelPath + File.separator + trial);
		if (!directory.exists()) {
			directory.mkdir();
			System.out.println("Directory: " + trial + " created");
		}

		ExperimentParameters exp_params = new ExperimentParameters(data_name, exp_file_setup);
		//ExperimentParameters exp_params = new ExperimentParameters(data_name, problem_type, should_normalize_data, scale, param_opt, directory, trial);
		TrainTree train = new TrainTree();

		System.out.println("BNerualT optimization:");

		NeuralTree n_tree = new NeuralTree();
		///* Generate controlled tree parameters size
		while(true) {
			System.out.print("Generating tree: ["+exp_params.n_max_children+" "+exp_params.n_max_depth+"] ");
			n_tree.genrateGenericRandomTree(exp_params);
			int tree_size =  n_tree.getTreeSize();
			if(!exp_params.n_bound_tree_size) {
				System.out.println("Ad-hoc Tree :"+tree_size);
				break;
			}
			if(tree_size > exp_params.n_min_tree_size_value) {
				System.out.println("Ad-hoc Tree (C) :"+tree_size);
				break;
			}
		}
		n_tree.saveTreeModel(n_tree, directory, (trial + "_Original"), exp_params.n_max_target_attr, exp_params.n_data_input_names, exp_params.n_data_target_names, true);
		train.parameterOptimization(exp_params, n_tree.copy_Tree(), null, directory, trial);
		// Finish of the tree experiments */
		System.out.println("Tree Experiments Ends..." + itr);
	}

	public static void main(String[] args) throws IOException {

		File exp_setup_path = new File(System.getProperty("user.dir"));
		String exp_file_name = "experiment_training_setup.txt";
		LinkedHashMap<String, Object> exp_file_setup = ReadExperiment.readExperimentSettingFromFile(exp_setup_path, exp_file_name);

		int  num_epp = (int) exp_file_setup.get("n_num_exp");
		String data_name = (String) exp_file_setup.get("n_data_name");

		System.out.println("Loading experiment file : "+exp_setup_path+File.separator+exp_file_name);
		System.out.println("	Num of times exp to repeat  : "+num_epp);
		System.out.println("	Data name                   : "+data_name);

		for(int i = 0; i < num_epp; i++) {
			System.out.println("experiment instance :"+i);
			runExperimentNew(i, data_name, exp_file_setup);
		}
		System.out.println("All experiments Ends...");
	}

}
