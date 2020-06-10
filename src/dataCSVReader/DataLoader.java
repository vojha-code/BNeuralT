/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataCSVReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import experimentSettings.ExperimentParameters;



/**
 *
 * @author yl918888
 */
public class DataLoader {

	public Data readCSVFile(String dataset_name, String problem_type, boolean noralized_data, double[] scale) throws IOException {
		System.out.println("Loading " + dataset_name + " dataset....");

		Data dataObj = null;
		try {
			long start = System.currentTimeMillis();
			DataPreporcessor dp = new DataPreporcessor();
			String fileName = ExperimentParameters.dataPath + File.separator + dataset_name;
			FileReader fin1 = new FileReader(fileName); // read lines of a csv file
			BufferedReader br1 = new BufferedReader(fin1);
			// ArrayList<String[]> listExamples = new ArrayList<String[]>();
			String line;
			int rows = 0, cols = 0;
			while ((line = br1.readLine()) != null) {
				//String[] tokens = line.split(",");
				if (rows == 0) {
					cols = line.split(",").length;
				}
				//listExamples.add(tokens);
				rows++;
			} // end file reading
			br1.close();
			fin1.close();

			FileReader fin2 = new FileReader(fileName); // read lines of a csv file
			BufferedReader br2 = new BufferedReader(fin2);
			// processing the ArrayList Data
			//rows = listExamples.size(); // number of columns
			//cols = ((String[]) listExamples.get(0)).length; // number of rows
			// System.out.print(rows);

			Object[][] dataset = new Object[rows][cols];
			// Object[] dataset = (Object[])listExamples.toArray();

			for (int i = 0; i < rows; i++) {
				//String[] rowValue = (String[]) listExamples.get(i);
				String[] rowValue = br2.readLine().split(",");
				//listExamples.set(i, null);
				for (int j = 0; j < cols; j++) {
					if (dp.isNumeric(rowValue[j])) {
						dataset[i][j] = Double.parseDouble(rowValue[j]);// Store Numeric value
					} else {
						dataset[i][j] = rowValue[j];// Store string value
					}
				}
				// System.out.println(i + ".");
			} // End collection of data
			br2.close();
			fin2.close();
			//listExamples = null; // free memory
			//System.gc();
			// System.out.println("Copied2");

			// Check feature type
			String[] features_names = new String[cols - 1];
			boolean[] string_type = new boolean[cols];
			//System.out.println(" ");
			for (int i = 0; i < (int) (rows * 0.1); i++) {// check only first 10 percent rows
				for (int j = 0; j < cols; j++) {
					Object dataVal = dataset[i][j];
					if (i == 0) {
						if (j != cols - 1) {
							features_names[j] = dataVal.toString();// only input feature names
						}
					} else {
						if (string_type[j] != true && !(dataVal instanceof Number)) {
							string_type[j] = true;
						}
						// System.out.print(" " + dataset[i][j]);
					}
				}
				//System.out.println(i + ".");
			} // End feature type checking
			if (problem_type.equalsIgnoreCase("Classification")) {
				string_type[cols - 1] = true;
			}

			// Collecting into inputs and targets values
			double[][] inputs = new double[rows - 1][cols - 1];
			double[][] targets = new double[rows - 1][1];
			String[] taget_names = null;
			for (int j = 0; j < cols; j++) {// take all columns
				//System.out.println(" " + string_type[j]);
				if (string_type[j]) {
					ArrayList<String> strList = new ArrayList<String>();
					for (int i = 1; i < rows; i++) {// starts from 1 because first (0th) row is names
						strList.add(dataset[i][j].toString());
					}
					List<String> list = new ArrayList<>(new HashSet<>(strList));
					Collections.sort(list);
					//System.out.println(list);
					if (j == cols - 1) {// for output/target column
						taget_names = list.toArray(new String[0]);
					}
					for (int i = 1; i < rows; i++) {// i starts from 1 because first row is feature name
						if (j != cols - 1) {
							inputs[i - 1][j] = list.indexOf(dataset[i][j].toString());
						} else {
							targets[i - 1][0] = list.indexOf(dataset[i][j].toString());
						}
					} // end for i
				} else {
					for (int i = 1; i < rows; i++) {// i starts from 1 because first row is feature name
						if (j != cols - 1) {
							inputs[i - 1][j] = Double.parseDouble(dataset[i][j].toString());
						} else {
							targets[i - 1][0] = Double.parseDouble(dataset[i][j].toString());
						}
					} // end for i
				}
			} // data set loaded to inputs and targets
			dataset = null;
			System.gc();

			/*System.out.print("Before Normalisation");
			dp.displayMatrix(inputs);
			dp.displayVector(dp.min(inputs, 0));
			dp.displayVector(dp.max(inputs, 0));
			dp.displayMatrix(targets);
			dp.displayVector(dp.min(targets, 0));
			dp.displayVector(dp.max(targets, 0));*/
			double[] features_min = null;
			double[] features_max = null;
			double[] target_min = null;
			double[] target_max = null;
			int axis = 0; //  columns min/max

			if (noralized_data) {
				features_min = dp.min(inputs, axis);// Inputs min- max
				features_max = dp.max(inputs, axis);// Inputs min- max
				System.out.println("   Data normalisation: scale [" + scale[0] + "," + scale[1] + "]");
				if (dataset_name.equalsIgnoreCase("mnist.csv")) {
					inputs = dp.nromInage(inputs, 256.0);
				} else {
					inputs = dp.normlizationMinMax(inputs, dp.min(inputs, 0), dp.max(inputs, 0), scale[0], scale[1]);
				}
			}
			//even if inputs are not normalised for regression problem normalise the output column
			if (!problem_type.equalsIgnoreCase("Classification")) {
				target_min = dp.min(targets, axis);
				target_max = dp.max(targets, axis);
				targets = dp.normlizationMinMax(targets, dp.min(targets, 0), dp.max(targets, 0), scale[0], scale[1]);
			}

			/*System.out.print("After Normalisation");
			dp.displayMatrix(inputs);
			dp.displayVector(dp.min(inputs, 0));
			dp.displayVector(dp.max(inputs, 0));
			dp.displayMatrix(targets);
			dp.displayVector(dp.min(targets, 0));
			dp.displayVector(dp.max(targets, 0));*/

			// For problem_type = "Classification";
			// Perform OneHotVector Encoding
			if (problem_type.equalsIgnoreCase("Classification")) {
				double[][] targetsOrignal = targets;
				targets = new double[rows - 1][taget_names.length];
				for (int i = 0; i < rows - 1; i++) {
					//System.out.print(i + " : ");
					for (int j = 0; j < taget_names.length; j++) {
						if ((int) targetsOrignal[i][0] == j) {
							targets[i][j] = 1;
						}
						//System.out.print(" " + (int) targets[i][j]);
					}
					//System.out.println("->" + (int) targetsOrignal[i][0] + "->" + taget_names[(int) targetsOrignal[i][0]]);
				}
				targetsOrignal = null;
				System.gc();
			} else {
				taget_names = new String[1];
				taget_names[0] = "target";// for regression there is only one target value
			} // End processing OneHotVectorEncoding
			// Loading data
			dataObj = new Data(dataset_name, problem_type, inputs, features_names, targets, taget_names, features_min, features_max, target_min, target_max, noralized_data);
			System.out.println("   Number of examples: " + inputs.length);
			System.out.println("   Number of inputs  : " + features_names.length);
			if (features_names.length > 10) {
				System.out.println("                       ...too many to display");
			} else {
				for (int j = 0; j < features_names.length; j++) {
					System.out.println("                     : " + features_names[j]);
				}
			}
			System.out.println("   Number of outputs : " + taget_names.length);
			for (int k = 0; k < taget_names.length; k++) {
				System.out.println("                     : " + taget_names[k]);
			}
			//dp.displayData(problem_type,inputs, features_names, targets, taget_names);
			inputs = null;
			targets = null;
			System.gc();

			long finish = System.currentTimeMillis();
			long timeElapsed = finish - start;
			System.out.println("Data Loading time: " + timeElapsed / 1000F + " sec.");
		} catch (IOException ex) {
			Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
		} //
		return dataObj;
	}// End of read File Method

	public static void main(String[] args) throws IOException {

		//All Tested OK
		//Testing and Validating Code for data loader
		DataLoader dl = new DataLoader();
		boolean normalize_data = true;
		double[] scale = { 0.0, 1.0 };

		String problem_type = "Classification";
		// Regression Problems
		//problem_type = "Regression";
		String[] dataNames = { "iris.csv", "wdbc.csv", "wine.csv", "heart.csv", "ionosphere.csv", "australian.csv", "pima.csv", "glass.csv", "vehicle.csv", "mnist.csv" };
		//String[] dataNames = { "dee.csv", "diabetese.csv", "baseball.csv", "friedman.csv", "mpg6.csv" };

		Data dataset = dl.readCSVFile(dataNames[0], problem_type, normalize_data, scale);

		Integer[] seq = new Integer[dataset.inputs.length];
		for (int i = 0; i < dataset.inputs.length; i++) {
			seq[i] = i;
			System.out.println(i + ":" + Arrays.toString(dataset.inputs[i]) + " " + Arrays.toString(dataset.target[i]));
		}
		//seq[1] = 0;

		DataPartitioner cv = new DataPartitioner(seq);

		/*HashMap<String, Object> dataMap = cv.holdout_method(dataset, 0.8, 0.0);
		double[][] train_In = (double[][]) dataMap.get("train_in");
		double[][] train_Tr = (double[][]) dataMap.get("train_tr");
		//Test
		double[][] test_In = (double[][]) dataMap.get("test_in");
		double[][] test_Tr = (double[][]) dataMap.get("test_tr");
		//Val
		double[][] val_In = (double[][]) dataMap.get("val_in");
		double[][] val_Tr = (double[][]) dataMap.get("val_tr");
		//Rand seq
		Integer[] rand_seq = (Integer[]) dataMap.get("rand_seq");
		System.out.println("\nCheck:  retirvied data: " + Arrays.toString(train_In[0]) + " = " + Arrays.toString(train_Tr[0]));
		System.out.println("\nCheck:  retirvied data: " + Arrays.toString(train_In[1]) + " = " + Arrays.toString(train_Tr[1])); */

		HashMap<String, Object> dataMap1 = cv.k_fold(dataset, 10);
		ArrayList<double[][]> inputFoldList = (ArrayList<double[][]>) dataMap1.get("input_folds");
		ArrayList<double[][]> targetFoldList = (ArrayList<double[][]>) dataMap1.get("target_folds");
		Integer[] rand_seq1 = (Integer[]) dataMap1.get("rand_seq");
		double[][] in_fold_idx = inputFoldList.get(0);
		double[][] tr_fold_idx = targetFoldList.get(0);
		System.out.println("\nCheck:  retirvied data: " + Arrays.toString(in_fold_idx[0]) + " " + Arrays.toString(tr_fold_idx[0]));
		System.out.println("\nCheck:  retirvied data: " + Arrays.toString(in_fold_idx[0]) + " " + Arrays.toString(tr_fold_idx[0]));
	}// End Main
}// End of class Data Loader
