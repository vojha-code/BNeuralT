package dataCSVReader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class DataPartitioner {

	Integer[] givenDataSequence;

	public DataPartitioner(Integer[] data_seq) {
		this.givenDataSequence = data_seq;
	}

	/**
	 * Spliting data using hold-out method
	 *
	 * @param data
	 * @param training_set % of total examples
	 * @param val_set      (% of reaming examples after training), i.e. division for
	 *                     remaining set in val and test
	 * @return
	 */
	public HashMap<String, Object> holdout_method(Data data, double training_set, double val_set) {
		System.out.println("\nPartitioning data using hold out method:");
		int examples = data.inputs.length;
		int features = data.inputs[0].length;
		int outputs = data.target[0].length;

		Integer[] random_sequence = new Integer[examples];
		// Randomising the data samples
		if (data.data_name.equalsIgnoreCase("mnist.csv")) {
			// Do not randomised the data
			training_set = 0.8571428571428571;
			random_sequence = genrateDataSquence(examples, false);
		} else {
			System.out.println("Shuffling (generating a random sequence..)");
			random_sequence = genrateDataSquence(examples, true);
			//System.out.println("\n" + Arrays.toString(random_sequence));
		}

		System.out.println("Spliting data into training, validation, and test sets:");
		// Spliting into training test and validation sets
		//Preparing training, validation, and test sets
		int trainset = (int) (examples * training_set);
		int valset = (int) ((examples - trainset) * val_set);
		int testset = (examples - trainset - valset);
		double[][] data_inputs_train = new double[trainset][features];
		double[][] data_targets_train = new double[trainset][outputs];

		double[][] data_inputs_test = new double[testset][features];
		double[][] data_targets_test = new double[testset][outputs];

		double[][] data_inputs_val = new double[valset][features];
		double[][] data_targets_val = new double[valset][outputs];

		for (int i = 0; i < examples; i++) {
			double[] inputs_vec = data.inputs[random_sequence[i]];
			double[] targets_vec = data.target[random_sequence[i]];
			if (i < trainset) {
				for (int j = 0; j < features; j++) {
					data_inputs_train[i][j] = inputs_vec[j];
				}
				for (int k = 0; k < outputs; k++) {
					data_targets_train[i][k] = targets_vec[k];
				}
			} else if (i < trainset + testset) {
				for (int j = 0; j < features; j++) {
					data_inputs_test[i - trainset][j] = inputs_vec[j];
				}
				for (int k = 0; k < outputs; k++) {
					data_targets_test[i - trainset][k] = targets_vec[k];
				}
			} else {
				for (int j = 0; j < features; j++) {
					data_inputs_val[i - trainset - testset][j] = inputs_vec[j];
				}
				for (int k = 0; k < outputs; k++) {
					data_targets_val[i - trainset - testset][k] = targets_vec[k];
				}
			}
		} //end for
		//Checking the percent of the split
		System.out.printf("    Training set   :%5d   %3.2f %%  \n", data_inputs_train.length, (data_inputs_train.length * 100.00) / examples);
		System.out.printf("    Test set       :%5d   %3.2f %%  \n", data_inputs_test.length, (data_inputs_test.length * 100.00) / examples);
		//System.out.printf("    Val set        :%5d   %3.2f %%  \n", data_targets_val.length, (data_targets_val.length * 100.00) / examples);
		System.out.printf("Data partition compleated!\n");
		//returning data split
		//ArrayList<Object> dataList = new ArrayList<Object>();
		HashMap<String, Object> dataMap = new HashMap<String, Object>();
		//Train
		dataMap.put("train_in", data_inputs_train);
		dataMap.put("train_tr", data_targets_train);
		//Test
		dataMap.put("test_in", data_inputs_test);
		dataMap.put("test_tr", data_targets_test);
		//Val
		dataMap.put("val_in", data_inputs_val);
		dataMap.put("val_tr", data_targets_val);
		//Rand seq
		dataMap.put("rand_seq", random_sequence);
		return dataMap;
	}//end holdout

	/**
	 *
	 * @param data  Data
	 * @param folds int numbe of folads
	 * @return
	 */
	public HashMap<String, Object> k_fold(Data data, int folds) {
		if (folds == 2) {
			return holdout_method(data, 0.5, 0.0);
		}

		System.out.println("\nPartitioning data using k-fold method:");
		int examples = data.inputs.length;
		int features = data.inputs[0].length;
		int outputs = data.target[0].length;
		// Randomizing the data samples
		Integer[] random_sequence = genrateDataSquence(examples, true);
		//System.out.println("\n" + Arrays.toString(random_sequence));

		System.out.println("Spliting data into training, validation, and test sets:");
		// Spliting into training test and validation sets
		//Preparing folds
		int foldLength = examples / folds;//length of each fold
		System.out.println("     Each fold length : " + foldLength + " i.e., " + (foldLength * 100.00 / examples) + " % each");
		ArrayList<double[][]> data_inputs_folds = new ArrayList<double[][]>();
		ArrayList<double[][]> data_target_folds = new ArrayList<double[][]>();

		int fold = 0;
		int foldExamples = 0;
		double[][] data_inputs = new double[foldLength][features];
		double[][] data_targets = new double[foldLength][outputs];
		int idx = 0;
		int foldIdx = 0;
		for (; idx < examples; idx++) {
			double[] inputs_vec = data.inputs[random_sequence[idx]];
			double[] targets_vec = data.target[random_sequence[idx]];
			if (idx < (foldExamples + foldLength) && fold < folds - 1) {
				//System.out.println(idx+" < "+(foldExamples + foldLength));
				for (int j = 0; j < features; j++) {
					data_inputs[foldIdx][j] = inputs_vec[j];
				}
				for (int k = 0; k < outputs; k++) {
					data_targets[foldIdx][k] = targets_vec[k];
				}
				foldIdx++;
			} else {
				if (fold < folds - 1) {
					System.out.println("         fold " + fold + " length " + data_inputs.length + " added at indx: " + idx);
					data_inputs_folds.add(data_inputs);
					data_target_folds.add(data_targets);
					foldExamples += foldLength;
					data_inputs = new double[foldLength][features];
					data_targets = new double[foldLength][outputs];
					fold++;
					foldIdx = 0;
				} else {
					System.out.println("         .....last fold retival at indx: " + idx);
					break;
				}
			}
		}
		foldLength = examples - idx + 1;
		foldIdx = 0;
		data_inputs = new double[foldLength][features];
		data_targets = new double[foldLength][outputs];
		idx = idx - 1;
		for (; idx < examples; idx++) {
			double[] inputs_vec = data.inputs[random_sequence[idx]];
			double[] targets_vec = data.target[random_sequence[idx]];
			for (int j = 0; j < features; j++) {
				data_inputs[foldIdx][j] = inputs_vec[j];
			}
			for (int k = 0; k < outputs; k++) {
				data_targets[foldIdx][k] = targets_vec[k];
			}
			foldIdx++;
		}
		System.out.println("         fold " + fold + " length " + data_inputs.length + " added at indx: " + idx);
		data_inputs_folds.add(data_inputs);
		data_target_folds.add(data_targets);
		System.out.print("     All " + data_inputs_folds.size() + " folds are retrived.....");

		HashMap<String, Object> dataMap = new HashMap<String, Object>();
		//Train
		dataMap.put("input_folds", data_inputs_folds);
		dataMap.put("target_folds", data_target_folds);
		dataMap.put("rand_seq", random_sequence);
		return dataMap;
	}//k-folds ends

	private Integer[] genrateDataSquence(int examples, boolean isRandom) {
		Integer[] newDataSequence = new Integer[examples];
		if (givenDataSequence == null) {
			for (int i = 0; i < examples; i++) {
				newDataSequence[i] = i;/// a non NON-random sequence
			}
			if (isRandom) {
				List<Integer> intList = Arrays.asList(newDataSequence);
				Collections.shuffle(intList);
				newDataSequence = intList.toArray(newDataSequence);
			}
			return newDataSequence;
		} else {// randomised data sequence only it was null
			System.out.println("Using the provided data squense for partition..");
			if (givenDataSequence.length != examples) {
				System.out.println("Wrong squence:  Seq MUST be equal to the number of total examples in input data!");
				System.exit(0);
			}
			List<Integer> intList = Arrays.asList(givenDataSequence);
			List<Integer> list = new ArrayList<>(new HashSet<>(intList));
			if (list.size() != examples) {
				System.out.println("Wrong squence: Each value in Seq MUST have unque!");
				System.exit(0);
			}
			return givenDataSequence;
		}
	}

}//End class
