package trainAndEvaluateTree;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.stream.IntStream;

import dataCSVReader.DataPartitioner;
import experimentSettings.ExperimentParameters;
import experimentSettings.ReadExperiment;
import neuralTree.EvaluateTree;
import neuralTree.NeuralTree;

public class EvaluateTreeModels {

	public static void main(String args[]) throws IOException {
		System.out.println("Tree evaluation modules");
		//File exp_setup_path = new File(System.getProperty("user.dir"));
		//LinkedHashMap<String, Object> exp_file_setup = ReadExperiment.readExperimentSettingFromFile(exp_setup_path, exp_file_name);
		//String filePathIn = (String) exp_file_setup.get("n_model_dir_path");
		//String evalTypeIn = (String) exp_file_setup.get("n_evaluation_type");

		String filePath = filePath = System.getProperty("user.dir") + File.separator + "trained_models" + File.separator;

		String evalType = "class_reg"; // "Classification" "Regression";
		if (args.length != 0) {
			if(args[0].equalsIgnoreCase("mnist")) {
				evalType = "mnist";
			}
		}

		//TODO:  Set these values

		String problem_type = "Classification"; // "Classification" "Regression";
		String[] data_names_calss_reg = { "iris.csv", "wdbc.csv", "wine.csv", "heart.csv", "ionosphere.csv", "australian.csv", "pima.csv", "glass.csv", "vehicle.csv","dee.csv", "mpg6.csv", "diabetese.csv", "baseball.csv", "friedman.csv"};
		String[] data_names_ptrn = { "mnist.csv"};

		//String collect_directory_res =  "File,Data,Algo,Size,Param,AccTst,TstTime,FunEvalTime\n";
		String  collect_directory_res =  "File,Data,Algo,Size,Fun,Leaf,Param,ETst,TstTime,FunEvalTime\n";
		//String  collect_directory_res =  "File,Data,Algo,Size,Fun,Leaf,Param,Etrn,ETst,RTrn,RTst,TrnTime,TstTime,FunEvalTime\n";

		File fileDirectory = null;
		File[] listOfFiles = null;
		File summaryFile = null;

		String[] data_name_list = null;

		if(evalType.equalsIgnoreCase("class_reg")) {
			fileDirectory = new File(filePath+ "pre_trained_class_reg_models");
			boolean collectExpermentOnly = false; // false for a report of time and param collection
			listOfFiles = fileDirectory.listFiles();

			data_name_list = data_names_calss_reg;
			summaryFile = new File(fileDirectory+File.separator+".."+File.separator+"BNeuralT_"+fileDirectory.getName()+"_coll.csv");
		}else {
			//collect_directory_res =  "File,Data,Algo,Size,Param,ErrTst\n";
			System.out.println("Loading MNIST data 70,000 examples (wait 8GB RMA: 50-70 seconds..)");
			System.out.println("Data first 60,000 examples are training and last 10,000 are test..");
			fileDirectory = new File(filePath+ "pre_trained_mnist_models_tab2");
			boolean collectExpermentOnly = false; // false for a report of time and param collection
			listOfFiles = fileDirectory.listFiles();

			data_name_list = data_names_ptrn;
			summaryFile = new File(fileDirectory+File.separator+".."+File.separator+"Table_2_BNeuralT_models.csv");
		}

		int count_files = 0;

		for(int didx = 0; didx < data_name_list.length ; didx++) {
			String data_name = data_name_list[didx];
			if(didx >  8) {
				problem_type =  "Regression";
			}

			File directory = null;
			String trial = "";
			String param_opt = "gd";
			boolean should_normalize_data = true;// TODO:  always check these parameters
			double[] scale = { 0.0, 1.0 };
			ExperimentParameters exp_params = new ExperimentParameters(data_name, problem_type, should_normalize_data, scale, param_opt, directory, trial);


			File[] listOfFilesL2 = null;
			if(evalType.equalsIgnoreCase("class_reg")) {
				for (int i = 0;  i <listOfFiles.length; i++) {
					if(listOfFiles[i].toString().contains(data_name.split("\\.")[0])){
						listOfFilesL2 = listOfFiles[i].listFiles();
					}
				}
			}else {
				listOfFilesL2 = listOfFiles;
			}
			
			//reading and collecting experiment results
			for(int j = 0; j < listOfFilesL2.length; j++) {
				if(listOfFilesL2[j].toString().contains(data_name.split("\\.")[0])){
					//System.out.println("Checking Directory: "+listOfFilesL2[j].toString());
					//File[] listOfModels = listOfFiles[i].listFiles();
					if(false) {
						String filename = listOfFilesL2[j].getName();
						//savedTreeExperimentFile(fileDirectory, filename, exp_params, ev, problem_type, train_Tr.length, test_Tr.length);
					}else{
						String loopcollect = getSavedTreeReports(listOfFilesL2[j], data_name.split("\\.")[0], exp_params, problem_type, evalType);
						if(!loopcollect.contentEquals(" ")) {
							collect_directory_res = collect_directory_res + loopcollect;
							count_files = count_files + 1;
						}
					}
				}else {
					System.out.println("data and directory name does not match"+listOfFilesL2[j]);
				}
			}// all inner file
		}//all_data
		System.out.println("Total Directoy available:  "+listOfFiles.length+" : directy read:  "+ count_files);
		if(count_files > 0) {
			try {
				FileWriter myWriter = new FileWriter(summaryFile);
				myWriter.write(collect_directory_res);
				myWriter.close();
				System.out.println(summaryFile.getName()+" successfully writtn to a file.");
				System.out.println("Location :"+(fileDirectory+File.separator+".."+File.separator));
			} catch (IOException e) {
				System.out.println("An error occurred.");
				e.printStackTrace();
			}
		}
	}//end of model property collection

	private static String savedTreeExperimentFile(File fileDirectory, String fileName, ExperimentParameters exp_params, EvaluateTree ev,String problem_type, int lengthTrn,int lengthTst) throws IOException {
		NeuralTree f_tree = ReadExperiment.getSavedTrees(fileDirectory, fileName);
		ev.set_dataset_to_evaluate("train");
		long start = System.currentTimeMillis();
		ev.getTreePredictedOutputsParallel(f_tree);
		long finish = System.currentTimeMillis();
		double timeTrain = (finish - start) / 1000.0;
		ArrayList<Object> errTrn =  ev.getTreeFitness();
		errTrn.add(f_tree.getTreeSize());
		//System.out.println("   Train Error : "+errTrn.get(0) +" time all: "+timeTrain + " time avg"+ timeTrain/lengthTrn);

		ev.set_dataset_to_evaluate("test");
		start = System.currentTimeMillis();
		ev.getTreePredictedOutputsParallel(f_tree);
		finish = System.currentTimeMillis();
		double timeTst = (finish - start) / 1000.0;
		//System.out.println("   Test Error : "+ev.getTreeFitness().get(0)+" "+timeTst+ " time avg"+ timeTst/lengthTst);
		ArrayList<Object> errTst =  ev.getTreeFitness();

		ArrayList<Object> listSavePerfroance = new ArrayList<Object>();
		listSavePerfroance.add(errTrn); // Save Best Structure Training Error
		listSavePerfroance.add(errTst);// Save Best Structure Test Error
		listSavePerfroance.add(IntStream.range(0, 20)); // Save Data Random Sequence
		listSavePerfroance.add(timeTrain/lengthTrn); // Save time elapsed
		exp_params.saveExperinetParameter(fileDirectory, "experiment_"+fileName.split("\\.")[0], "_tst", listSavePerfroance);
		return null;
	}

	private static String getSavedTreeReports(File fileDirectory, String data_name, ExperimentParameters exp_params,  String problem_type, String evalType) throws IOException {

		Integer[] training_data_sequence =  null;


		String fileName = "";// "mnist_0_2020_05_02_15_58_44_rmsprop_optPRM_model.json";
		String dataName = null;
		// Get only tree model and read parameter from setting

		ArrayList<NeuralTree> f_treeList = new ArrayList<NeuralTree>();
		ArrayList<String> f_treeListName = new ArrayList<String>();

		if(false) {
			f_treeList.add(ReadExperiment.getSavedTrees(fileDirectory, fileName));
		}else {
			///* Get Entire experiment setting and read random sequence
			ArrayList<Object> rtnList = ReadExperiment.getSavedExperiment(fileDirectory, "all");
			//System.out.println("\n\n"+rtnList.size());
			if(((ArrayList<NeuralTree>) rtnList.get(0)).size() == 0){
				System.out.println("Directory does not have the required model file");
			}else {
				f_treeList = ((ArrayList<NeuralTree>)rtnList.get(0));
				f_treeListName = ((ArrayList<String>)rtnList.get(2));
			}
			if(((ArrayList<LinkedHashMap<String, Object>>)rtnList.get(1)).size() == 0) {
				//System.out.println("Directory does not have OR ");
			}else {
				ArrayList<LinkedHashMap<String, Object>> hashmapList = (ArrayList<LinkedHashMap<String, Object>>)rtnList.get(1);
				//Reading parameters of the experiment setting
				dataName =(String)hashmapList.get(0).get("n_data_name");
				System.out.println(dataName);

				int[] seq = (int[])hashmapList.get(0).get("n_data_rand_secquence");
				training_data_sequence = Arrays.stream(seq).boxed().toArray(Integer[]::new);
				System.out.println("Retrieved Experiment Data Seq.");
				//System.out.println("Retrieved Experiment Data Seq: "+Arrays.deepToString(training_data_sequence));
			}
		}

		DataPartitioner cv = new DataPartitioner(training_data_sequence);
		HashMap<String, Object> dataMap = cv.holdout_method(exp_params.dataset, 0.8, 0.0);
		double[][] train_In = (double[][]) dataMap.get("train_in");
		double[][] train_Tr = (double[][]) dataMap.get("train_tr");
		//Test
		double[][] test_In = (double[][]) dataMap.get("test_in");
		double[][] test_Tr = (double[][]) dataMap.get("test_tr");

		EvaluateTree ev = new EvaluateTree(train_In, train_Tr, test_In, test_Tr, null, null, exp_params.n_data_target_min, exp_params.n_data_target_max, exp_params.n_scale, problem_type);

		String  collect =  "";
		//String  collect =  "File,Size,Fun,Leaf,Param,Etrn,ETst,RTrn,RTst,TrnTime,TstTime,FunEvalTime\n";
		for (int i =0; i< f_treeList.size();i++) {
			String loopCollect = "";
			NeuralTree f_tree = f_treeList.get(i);
			String treeFile = f_treeListName.get(i);

			int treeSize = f_tree.getTreeSize();
			int treeFunNode = f_tree.getFuncNodeSize();
			int treeLeafNode = f_tree.getLeafNodeSize();
			int treeParam = treeFunNode*2 + treeLeafNode;
			/*String[] algo = {"gd","momentum_gd","nesterov_accelerated_gd","adagrad","rmsprop","adam"};
			String [] treeFileSplit = treeFile.split("_");
			String algoVal = "";
			for(String aval: algo) {
				for(String sval: treeFileSplit) {
					if(sval.equalsIgnoreCase(aval)) {
						algoVal = aval;
						break;
					}
				}
			} */
			String algoVal = (treeFile.split("optPRM")[1]).split("model")[0];
			//loopCollect = loopCollect + treeFile+","+data_name+","+algoVal+","+treeSize+","+treeParam+",";
			loopCollect = loopCollect + treeFile+","+data_name+","+algoVal+","+treeSize+","+treeFunNode+","+treeLeafNode+","+treeParam+",";
			//loopCollect = loopCollect + treeFile+","+data_name+","+algoVal+","+treeSize+","+treeFunNode+","+treeLeafNode+","+treeParam+",";

			//System.out.println("Tree: "+treeFile);
			//System.out.println("   Size: "+treeSize);
			//System.out.println("   Fun: "+treeFunNode);
			//System.out.println("   Leaf: "+treeLeafNode);
			//System.out.println("   Param: "+treeParam);

			/*ev.set_dataset_to_evaluate("train");
			long start = System.currentTimeMillis();
			ev.getTreePredictedOutputsParallel(f_tree);
			long finish = System.currentTimeMillis();
			double timeTrain = (finish - start) / 1000.0;
			ArrayList<Object> errTrn =  ev.getTreeFitness(); */
			//System.out.println("   Train Error : "+errTrn.get(0) +" time all: "+timeTrain + " time avg"+ timeTrain/lengthTrn);

			ev.set_dataset_to_evaluate("test");
			long start = System.currentTimeMillis();
			if(data_name.contains("mnist")) {
				ev.getTreePredictedOutputsParallel(f_tree);
			}else {
				ev.getTreePredictedOutputs(f_tree);
			}
			long finish = System.currentTimeMillis();
			double timeTst = (finish - start) / 1000.0;
			//System.out.println("   Test Error : "+ev.getTreeFitness().get(0)+" "+timeTst+ " time avg"+ timeTst/lengthTst);

			ArrayList<Object> errTst =  ev.getTreeFitness();
			if (problem_type.equalsIgnoreCase("Classification")){
				//loopCollect = loopCollect + errTrn.get(0)+","+errTst.get(0)+",--,--,";
				if(evalType.equalsIgnoreCase("class_reg")){
					loopCollect = loopCollect + (1.0 - (double)errTst.get(0))+","; // test error
				}else {
					loopCollect = loopCollect + (double)errTst.get(0)+","; // test error
					//loopCollect = loopCollect + (double)errTst.get(0); // test error
				}
			}else {
				//loopCollect = loopCollect + errTrn.get(0)+","+errTst.get(0)+","+errTrn.get(2)+","+errTst.get(2)+",";
				loopCollect = loopCollect + (double)errTst.get(2)+","; // r2 (coefficient of determinant)
				if(treeSize < 7) {// min tree threshold applied to regression problem for rejecting trees in 30 runs
					return collect = " ";
				}
			}
			//loopCollect = loopCollect + timeTrain+","+timeTst+","+(timeTrain/lengthTrn);
			if(evalType.equalsIgnoreCase("class_reg")){
				loopCollect = loopCollect + timeTst+","+(timeTst/test_Tr.length);
			}else {
				//Suppress if you do notwant MNIST time
				loopCollect = loopCollect + timeTst+","+(timeTst/test_Tr.length);
			}
			collect = collect + loopCollect +"\n";
		}
		return collect;
	}
}
