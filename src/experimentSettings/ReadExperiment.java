package experimentSettings;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Scanner;

import neuralTree.NeuralTree;

public class ReadExperiment {
	public static ArrayList<Object> getSavedExperiment(File fileDirectory, String readSavedModelType) throws IOException {

		ArrayList<Object> rtnList =  new ArrayList<Object>();
		ArrayList<NeuralTree> treeList =  new ArrayList<NeuralTree>();
		ArrayList<String> treeListFileName =  new ArrayList<String>();
		ArrayList<LinkedHashMap<String, Object>> paramterList =  new ArrayList<LinkedHashMap<String, Object>>();
		File[] listOfFiles = fileDirectory.listFiles();

		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile()) {
				String fileName = listOfFiles[i].getName();

				// check model files in the experiment folder
				if(fileName.contains("model.json")) {
					if(readSavedModelType.equalsIgnoreCase("Original")){
						if(fileName.contains("model.json") && fileName.contains("Original")){
							//System.out.println("File " + fileName);
							//String fileMdel = "vehicle_1_2020_05_24_09_31_39_Original_model.json";
							treeList.add(getSavedTrees(fileDirectory,fileName));
							treeListFileName.add(fileName);
						}
					}
					else if(readSavedModelType.equalsIgnoreCase("all")){
						//System.out.println("Checking all File " + fileName);
						if(fileName.contains("model.json")){
							//System.out.println("File " + fileName);
							//String fileMdel = "vehicle_1_2020_05_24_09_31_39_Original_model.json";
							treeList.add(getSavedTrees(fileDirectory,fileName));
							treeListFileName.add(fileName);
						}
					}
					else{
						//System.out.println("Checking nothing ..." + fileName);
					}
				}
				if(fileName.contains("experiment")) {
					//System.out.println("Checking experiment: found: " + fileName);
					LinkedHashMap<String, Object> hashmap =	readExperimentSettingFromFile(fileDirectory,fileName);
					paramterList.add(hashmap);
				}
			} else if (listOfFiles[i].isDirectory()) {
				System.out.println("Directory " + listOfFiles[i].getName());
			}
		}
		rtnList.add(treeList);
		rtnList.add(paramterList);
		rtnList.add(treeListFileName);
		return rtnList;
	}

	public static LinkedHashMap<String, Object> readExperimentSettingFromFile(File fileDirectory, String fileName) {
		LinkedHashMap<String, Object> hashmap = new LinkedHashMap<String, Object>();
		try {
			File expFile = new File(fileDirectory + File.separator + fileName);
			Scanner expFileScanner = new Scanner(expFile);
			while (expFileScanner.hasNextLine()) {
				String line = expFileScanner.nextLine();
				line = line.replace("{", "");
				line = line.replace("\"", "");
				//System.out.println(line);
				String[] paramter =  line.split(":");
				if(paramter.length == 2) {
					if(paramter[1].contains("[")) { // Parameter is an array
						//System.out.print(" "+paramter[0]+":"+paramter[1]);
						paramter[1] = paramter[1].replace("[", "");
						paramter[1] = paramter[1].replace("]", "");
						paramter[1] = paramter[1].replace(" ", "");
						String[] arrayS =  paramter[1].split(",");
						if(isNumeric(arrayS[0])) {
							//System.out.print(" -> array: ");
							if(arrayS[0].contains(".") && isNumeric(arrayS[0])) {
								//System.out.print(" -> double");
								double[] arrayD = new double[arrayS.length];
								for(int i = 0; i< arrayS.length;i++) {
									arrayD[i] = Double.parseDouble(arrayS[i]);
								}
								hashmap.put(paramter[0], arrayD);
							}else {
								//System.out.print("-> int");
								int[] arrayI = new int[arrayS.length];
								for(int i = 0; i< arrayS.length;i++) {
									arrayI[i] = Integer.parseInt(arrayS[i]);
								}
								//System.out.println("print"+paramter[0]);
								hashmap.put(paramter[0], arrayI);
							}
						}else {
							//System.out.print(" -> string");
							hashmap.put(paramter[0], arrayS);
						}
						//System.out.println(" : "+arrayS[0]);
					}else { // Parameter is not an array
						paramter[1] = paramter[1].replace(" ", "");
						String[] arrayS =  paramter[1].split(",");
						//System.out.print(" "+paramter[0]+" : "+arrayS[0]);
						if(isNumeric(arrayS[0])) {
							//System.out.print(" -> numberic Value ");
							if(arrayS[0].contains(".")) {
								//System.out.print(" -> double ");
								double value  = Double.parseDouble(arrayS[0]);
								hashmap.put(paramter[0], value);
							}else {
								//System.out.print(" -> int ");
								int value  = Integer.parseInt(arrayS[0]);
								hashmap.put(paramter[0], value);
							}
						}else {// Parameter is a String
							if(arrayS[0].equalsIgnoreCase("true") || arrayS[0].equalsIgnoreCase("false")) {
								//System.out.print(" -> bool : "+Boolean.parseBoolean(arrayS[0]));
								hashmap.put(paramter[0], Boolean.parseBoolean(arrayS[0]));
								//}if(arrayS[0].equalsIgnoreCase("false")) {
								//	System.out.print(" -> bool : "+arrayS[0]);
								//	hashmap.put(paramter[0], new Boolean("false"));
							}else {
								//System.out.print(" -> string ");
								hashmap.put(paramter[0], arrayS[0]);
							}
						}
						//System.out.println(" : "+arrayS[0]);
					}
				}
			}
			expFileScanner.close();
		} catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		//int[] dataSeq = (int[])hashmap.get("n_data_rand_secquence");
		//System.out.println(Arrays.toString(dataSeq));
		return hashmap;
	}

	public static boolean isNumeric(String str) {
		return str.matches("-?\\d+(\\.\\d+)?");  //match a number with optional '-' and decimal.
	}


	public static NeuralTree getSavedTrees(File fileDirectory, String fileName) {
		NeuralTree file_tree = new NeuralTree();
		file_tree.readTreeModel(fileDirectory, fileName);
		//System.out.println(file_tree.getTreeSize());
		return file_tree;
	}
}
