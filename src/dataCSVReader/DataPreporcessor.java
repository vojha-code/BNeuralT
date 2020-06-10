package dataCSVReader;

public class DataPreporcessor {
	/**
	 *
	 * @param matrix double[][] 2D matrix
	 * @param axis   int 0 = column-wise or 1 = row-wise
	 * @return double[] max
	 */
	public double[] max(double[][] matrix, int axis) {
		int axixLength;
		int cols = matrix[0].length; // number of columns
		int rows = matrix.length; // number of rows

		if (axis == 0) {
			axixLength = cols;
		} else {
			axixLength = rows;
		}

		double[] max = new double[axixLength];
		if (axis == 0) {
			for (int j = 0; j < cols; j++) {
				max[j] = matrix[0][j];// search for each columns
				for (int i = 0; i < rows; i++) {
					if (matrix[i][j] > max[j]) { // if current cell is gretter than temp
						max[j] = matrix[i][j];
					}
				}
			}
		} else {
			for (int i = 0; i < rows; i++) {
				max[i] = matrix[i][0];// search for each rows
				for (int j = 1; j < cols; j++) {
					if (matrix[i][j] > max[i]) { // if current cell is gretter than temp
						max[i] = matrix[i][j];
					}
				}
			}
		}
		return max;
	}// Max calucation ends

	/**
	 *
	 * @param matrix double[][] 2D matrix
	 * @param axis   int 0 = column-wise or 1 = row-wise
	 * @return double[] min
	 */
	public double[] min(double[][] matrix, int axis) {
		int axixLength;
		int rows = matrix.length; // number of rows
		int cols = matrix[0].length; // number of columns

		if (axis == 0) {
			axixLength = cols;
		} else {
			axixLength = rows;
		}

		double[] min = new double[axixLength];
		if (axis == 0) {
			for (int j = 0; j < cols; j++) {
				min[j] = matrix[0][j];// search for each columns
				for (int i = 0; i < rows; i++) {
					if (matrix[i][j] < min[j]) { // if current cell is less than temp
						min[j] = matrix[i][j];
					}
				}
			}
		} else {
			for (int i = 0; i < rows; i++) {
				min[i] = matrix[i][0];// search for each rows
				for (int j = 1; j < cols; j++) {
					if (matrix[i][j] < min[i]) { // if current cell is less than temp
						min[i] = matrix[i][j];
					}
				}
			}
		}
		return min;
	}// Min calculation ends

	/**
	 *
	 * @param matrix double[][]: 2D matrix
	 * @param min    double[]: min of clos
	 * @param max    double[]: max of clos
	 * @param low    double: scaling low
	 * @param high   double: scaling high
	 * @return double[][] normalized matrix
	 */
	public double[][] normlizationMinMax(double[][] matrix, double[] min, double[] max, double low, double high) {
		int rows = matrix.length; // number of rows
		int cols = matrix[0].length; // number of columns
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = normalize(matrix[i][j], min[j], max[j], low, high);
			}
		}
		return matrix;
	}// end normalization

	/**
	 *
	 * @param matrix      2D matrix double[][]
	 * @param maxIntesity double
	 * @return matrix double[][]
	 */
	public double[][] nromInage(double[][] matrix, double maxIntesity) {
		int rows = matrix.length; // number of rows
		int cols = matrix[0].length; // number of columns
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = (matrix[i][j]) / maxIntesity;
			}
		}
		return matrix;
	}// end normalization

	/**
	 *
	 * @param matrix      2D matrix double[][]
	 * @param maxIntesity double
	 * @return matrix double[][]
	 */
	public double[][] deNormInage(double[][] matrix, double maxIntesity) {
		int rows = matrix.length; // number of rows
		int cols = matrix[0].length; // number of columns
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = (matrix[i][j]) * maxIntesity;
			}
		}
		return matrix;
	}// end normalization

	/**
	 *
	 * @param matrix double[][]: 2D normalized matrix
	 * @param min    double[]: min of clos
	 * @param max    double[]: max of clos
	 * @param low    double: scaling low
	 * @param high   double: scaling high
	 * @return double[][] de-normalized matrix
	 */
	public double[][] deNormlizationMinMax(double[][] matrix, double[] min, double[] max, double low, double high) {
		int rows = matrix.length; // number of rows
		int cols = matrix[0].length; // number of columns
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = denormalize(matrix[i][j], min[j], max[j], low, high);
			}
		}
		return matrix;
	}// end denormalization

	/**
	 * //Normalizing data value
	 *
	 * @param x              double: value to be normalized
	 * @param dataLow        double: min of column
	 * @param dataHigh       double: max of column
	 * @param normalizedLow  double: low of scale
	 * @param normalizedHigh double: high of scale
	 * @return double: normalized values
	 */
	public double normalize(double x, double dataLow, double dataHigh, double normalizedLow, double normalizedHigh) {
		return ((x - dataLow) / (dataHigh - dataLow)) * (normalizedHigh - normalizedLow) + normalizedLow;
	}// end normalizatioin single value

	/**
	 * //De Normalizing data value
	 *
	 * @param x              double: value to be normalized
	 * @param dataLow        double: min of column
	 * @param dataHigh       double: max of column
	 * @param normalizedLow  double: low of scale
	 * @param normalizedHigh double: high of scale
	 * @return double: de-normalized values
	 */
	public double denormalize(double x, double dataLow, double dataHigh, double normalizedLow, double normalizedHigh) {
		return ((dataLow - dataHigh) * x - normalizedHigh * dataLow + dataHigh * normalizedLow) / (normalizedLow - normalizedHigh);
	}// end de-normalizatioin single value

	// check data is numeric
	/**
	 *
	 * @param s String value
	 * @return boolean: true/false
	 */
	public boolean isNumeric(String s) {
		return s.matches("[-+]?\\d*\\.?\\d+");
	}

	/**
	 *
	 * @param matrix double[][]
	 */
	public void displayMatrix(double[][] matrix) {
		int cols = matrix[0].length; // number of columns
		int rows = matrix.length; // number of rows
		for (int i = 0; i < rows; i++) {
			System.out.printf(i + ": [");
			for (int j = 0; j < cols; j++) {
				printValue(matrix[i][j]);
			}
			System.out.println(" ]");
		}
	}//end print

	/**
	 *
	 * @param vector double[]
	 */
	public void displayVector(double[] vector) {
		int cols = vector.length; // number of rows
		System.out.printf("[");
		for (int j = 0; j < cols; j++) {
			printValue(vector[j]);
		}
		System.out.println(" ]");
	}//end print

	/**
	 * Dipslay data
	 *
	 * @param matrix double[][]
	 */
	public void displayData(String problem_type, double[][] inputs, String[] features_names, double[][] target, String[] taget_names) {
		System.out.println("Printing data...");
		int cols = inputs[0].length; // number of columns
		int rows = inputs.length; // number of rows
		int outputs = target[0].length;
		for (int i = 0; i < rows; i++) {
			System.out.printf("%5d : [", i);
			for (int j = 0; j < cols; j++) {
				printValue(inputs[i][j]);
			}
			System.out.printf(" : ");
			for (int k = 0; k < outputs; k++) {
				if (problem_type.equalsIgnoreCase("Classification")) {
					printValueInt((int) target[i][k]);
				} else {
					printValue(target[i][k]);
				}
			}
			System.out.println(" ]");
		}
	}

	public void printValue(double value) {
		if (value < 0.0) {
			System.out.printf(" %3.2f", value);
		} else {
			System.out.printf("  %3.2f", value);
		}
	}

	public void printValueInt(int value) {
		if (value < 0.0) {
			System.out.printf(" %3d", value);
		} else {
			System.out.printf("  %3d", value);
		}
	}
}
