package reporting;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import experimentSettings.ExperimentParameters;
import optimization.Individual;


public class SaveAlgorithmRun {

	/**
	 *
	 * @param mPopulation Individual[] population of
	 * @param pExpParams experiment parameters
	 * @param directory
	 * @param trial filename
	 * @param itr iteration number
	 * @param algo algorithm
	 * @param isFinal is its a the final population
	 */
	public static void saveGpPopulation(Individual[] mPopulation, ExperimentParameters pExpParams, File directory, String trial, String itr, String algo, boolean isFinal, boolean saveTree) {
		String plotName = "treeItrPopulation_";

		File directoryItr = new File(directory + File.separator + itr);
		if (!directoryItr.exists()) {
			directoryItr.mkdir();
		}
		String data_filepath = directoryItr + File.separator + plotName + algo + trial + "_objective.csv";
		FileWriter csvWriter;
		try {
			csvWriter = new FileWriter(data_filepath);
			// Save iteration_data
			int pop_index = 0;
			for (Individual pop : mPopulation) {
				csvWriter.append(pop.mCost[0] + "," + pop.mCost[1]);
				csvWriter.append("\n");
				if (saveTree) {
					pop.mTree.saveTreeModel(pop.mTree, directoryItr, (trial + "_" + pop_index), pExpParams.n_max_target_attr, pExpParams.n_data_input_names, pExpParams.n_data_target_names, isFinal);
				}
				pop_index++;
			}
			csvWriter.flush();
			csvWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void saveGpIteration(ArrayList<Object> performance_record, File directory, String trial, String algo, String pProblemType, int p_max_target_attr) {

		String data_filepath = directory + File.separator + "treeIteration_" + trial + "_values_" + algo + ".csv";
		FileWriter csvWriter;
		try {
			csvWriter = new FileWriter(data_filepath);
			if (pProblemType.equalsIgnoreCase("Classification")) {
				if (p_max_target_attr != ((double[]) ((ArrayList<Object>) performance_record.get(0)).get(1)).length) {
					System.out.println("Somthing wrong with Confusion Matrix of Algo perfroance record");
					System.exit(0);
				}
				String header = "error,";
				String[] headerValues = { "prec", "recall", "spec", "tp", "fp", "fn", "tn" };
				for (String colName : headerValues) {
					header = getStringHeaderClass(p_max_target_attr, header, colName);
				}
				header = header + "tree_size";
				csvWriter.append(header + "\n");

				//iterate through all iteration record.
				for (Object iterObj : performance_record) {
					ArrayList<Object> nCostAll = (ArrayList<Object>) iterObj;

					String itrValues = "" + (double) nCostAll.get(0) + ",";// error
					for (int i = 1; i < 4; i++) {
						itrValues = getStringClassLength((double[]) nCostAll.get(i), p_max_target_attr, itrValues);
					}
					for (int i = 4; i < 8; i++) {
						itrValues = getStringClassLength((int[]) nCostAll.get(i), p_max_target_attr, itrValues);
					}
					itrValues = itrValues + (int) nCostAll.get(8) + ",";// tree_size
					csvWriter.append(itrValues + "\n");
				}
			} // for classification
			else {
				// for regression
				String header = "error,";
				header = header + "corr,";
				header = header + "r2,";
				header = header + "tree_size";
				csvWriter.append(header + "\n");
				//iterate through all iteration record.
				for (Object iterObj : performance_record) {
					ArrayList<Object> nCostAll = (ArrayList<Object>) iterObj;
					String itrValues = "" + (double) nCostAll.get(0) + ",";// error
					itrValues = itrValues + (double) nCostAll.get(1) + ",";// r
					itrValues = itrValues + (double) nCostAll.get(2) + ",";// r2
					itrValues = itrValues + (int) nCostAll.get(3) + ",";// tree_size
					csvWriter.append(itrValues + "\n");
				}
			} //end class / regression
			csvWriter.flush();
			csvWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static String getStringHeaderClass(int len, String header, String colName) {
		for (int i = 0; i < len; i++) {
			header = header + colName + "_" + i + ",";
		}
		return header;
	}

	private static String getStringClassLength(double[] x, int len, String itrValues) {
		for (int i = 0; i < len; i++) {
			itrValues = itrValues + x[i] + ",";
		}
		return itrValues;
	}

	private static String getStringClassLength(int[] x, int len, String itrValues) {
		for (int i = 0; i < len; i++) {
			itrValues = itrValues + x[i] + ",";
		}
		return itrValues;
	}

	public static void saveSGDIteration(ArrayList<double[]> performance_record_stocstic, File directory, String trial, String algo) {
		String data_filepath = directory + File.separator + "sgdIteration_" + trial + "_values_"+algo+".csv";
		FileWriter csvWriter;
		try {
			csvWriter = new FileWriter(data_filepath);
			//iterate through all iteration record.
			for (double[] iterArray : performance_record_stocstic) {
				String itrValues = "";// error
				for (int i = 0; i < iterArray.length; i++) {
					itrValues = itrValues + iterArray[i] + ",";
				}
				csvWriter.append(itrValues + "\n");
			}
			csvWriter.flush();
			csvWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
