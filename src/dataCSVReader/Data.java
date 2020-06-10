/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataCSVReader;

/**
 *
 * @author yl918888
 */
public class Data {

	String problem_type;
	String data_name;
	boolean is_normlised;

	public double[][] inputs; // a 2D Matrix of data values training and text set combined
	public String[] feature_names;
	public double[] features_min; //  hold feature min value for min max normalisation
	public double[] features_max; //  hold feature max value for min max normalisation

	public double[][] target; // a 2D Matrix of target -  for Classification [samples x classes] and for regression [samples x 1]
	public double[] target_min;
	public double[] target_max;
	public String[] target_names; //  a vector of string to preserve target/class names

	/**
	 * @param dataName String: dataset name
	 * @param problemType String problem type e.g. classification regression
	 * @param pinputs double[][]: a Matrix of double of inputs values
	 * @param pfeature_names String[] a vector feature names
	 * @param ptarget double[][]: target
	 * @param ptarget_names target names
	 * @param noralized_data
	 * @param is_normlised
	 */
	public Data(String dataName, String problemType, double[][] pinputs, String[] pfeature_names, double[][] ptarget, String[] ptarget_names, double[] features_min2, double[] features_max2,
			double[] target_min2, double[] target_max2, boolean noralized_data) {
		this.problem_type = problemType;
		this.data_name = dataName;
		this.is_normlised = noralized_data;

		this.inputs = pinputs;
		this.feature_names = pfeature_names;

		this.target = ptarget;
		this.target_names = ptarget_names;

		if (this.is_normlised) {
			this.features_min = features_min2;
			this.features_max = features_max2;
		}
		if (!problem_type.equalsIgnoreCase("Classification")) {
			// if NOT classification
			this.target_min = target_min2;
			this.target_max = target_max2;
		}
	}

	public static void main(String[] args) {
		/*
		 * /Testing and Validating Code data processing
		Data dataObj = new Data();
		Object[][] data = {{"in1", "in2", "in3", "in4", "target"},
		{1, "A", 2, 3.5, "C1"},
		{5, "D", 2, -2, "C3"},
		{6, "A", -6.5, 89.6, "C3"},
		{3, "D", -6.5, 9.6, "C1"},
		{5, "C", -6.5, 5.6, "C2"}
		};
		int cols = data[0].length; // number of columns
		int rows = data.length; // number of rows
		String[] featuer_names = new String[cols];

		boolean[] string_type = new boolean[cols];
		for (int j = 0; j < cols; j++) {
		    System.out.print(" " + string_type[j]);
		}
		System.out.println(" ");
		for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < cols; j++) {
		        Object dataVal = data[i][j];
		        if (i == 0) {
		            featuer_names[j] = dataVal.toString();
		        } else {
		            if (string_type[j] != true && !(dataVal instanceof Number)) {
		                string_type[j] = true;
		            }
		            System.out.print(" " + data[i][j]);
		        }
		    }
		    System.out.println(" ");
		}

		double[][] inputs = new double[rows - 1][cols - 1];
		double[][] targets = new double[rows - 1][1];
		String[] taget_names = null;
		for (int j = 0; j < cols; j++) {
		    System.out.println(" " + string_type[j]);
		    if (string_type[j]) {
		        ArrayList<String> stringList = new ArrayList();
		        for (int i = 1; i < rows; i++) {
		            stringList.add(data[i][j].toString());
		        }
		        List<String> list = new ArrayList<>(new HashSet<>(stringList));
		        Collections.sort(list);
		        System.out.println(list);
		        if (j == cols - 1) {
		            taget_names = list.toArray(new String[0]);
		        }
		        for (int i = 1; i < rows; i++) {
		            if (j != cols - 1) {
		                inputs[i - 1][j] = list.indexOf(data[i][j].toString());
		            } else {
		                targets[i - 1][0] = list.indexOf(data[i][j].toString());
		            }
		        }
		    } else {
		        for (int i = 1; i < rows; i++) {
		            if (j != cols - 1) {
		                inputs[i - 1][j] = Double.parseDouble(data[i][j].toString());
		            } else {
		                targets[i - 1][0] = Double.parseDouble(data[i][j].toString());
		            }
		        }
		    }
		}

		System.out.println(" ");
		for (int i = 0; i < rows - 1; i++) {
		    for (int j = 0; j < cols - 1; j++) {
		        System.out.print(" " + inputs[i][j]);
		    }
		    System.out.println("->" + targets[i][0] + "->" + taget_names[(int) targets[i][0]]);
		}
		System.out.println();
		String problem = "Classification";
		if (problem.equalsIgnoreCase("Classification")) {
		    double[][] targetsOrignal = targets;
		    targets = new double[rows - 1][taget_names.length];
		    for (int i = 0; i < rows - 1; i++) {
		        for (int j = 0; j < taget_names.length; j++) {
		            if ((int) targetsOrignal[i][0] == j) {
		                targets[i][j] = 1;
		            }
		            System.out.print(" " + (int) targets[i][j]);
		        }
		        System.out.println("->" + (int) targetsOrignal[i][0] + "->" + taget_names[(int) targetsOrignal[i][0]]);
		    }
		}


		double[][] inputs = new Object[m][n];
		double[][] targets = new Object[m][1];


		for (int i = 0; i < m; i++) {
		    targets[i][0] = i;
		    for (int j = 0; j < n; j++) {
		        inputs[i][j] = Math.round(Math.random() * 10);
		    }
		}

		String[] feature_names = {"A", "B", "C"};
		String[] traget_names = {"t"};

		System.out.println(inputs[0].length); // number of columns
		System.out.println(inputs.length); // number of rows

		Data data = new Data(inputs, feature_names, targets, traget_names);
		double[] dataMin = data.min(inputs, 0);
		double[] dataMax = data.max(inputs, 0);
		data.displayVector(dataMin);
		data.displayVector(dataMax);

		System.out.println(" Original Matrix ");
		data.displayMatrix(inputs);

		System.out.println(" Min Max of Matrix ");
		data.displayVector(data.min(inputs, 0));
		data.displayVector(data.max(inputs, 0));

		double[][] inputsNorm = data.normlizationMinMax(inputs, dataMin, dataMax, 0.0, 1.0);
		System.out.println(" Normalized Matrix ");
		data.displayMatrix(inputsNorm);

		System.out.println(" Min Max of Matrix ");
		data.displayVector(data.min(inputsNorm, 0));
		data.displayVector(data.max(inputsNorm, 0));

		double[][] inputsDeNorm = data.deNormlizationMinMax(inputsNorm, dataMin, dataMax, 0.0, 1.0);
		System.out.println(" De Normalized Matrix ");
		data.displayMatrix(inputsDeNorm);

		System.out.println(" Min Max of Matrix ");
		data.displayVector(data.min(inputsDeNorm, 0));
		data.displayVector(data.max(inputsDeNorm, 0));*/
	}
}
