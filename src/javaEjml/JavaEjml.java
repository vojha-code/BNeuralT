package javaEjml;

import java.util.ArrayList;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

public class JavaEjml {
	/////////////////////////   1 D  ///////////////////////////////
	public static SimpleMatrix java_to_SimpleMatrix_1D(double[] vector) {
		SimpleMatrix sm1D = new SimpleMatrix(vector.length, 1);
		for (int i = 0; i < vector.length; i++) {
			sm1D.set(i, vector[i]);
		}
		return sm1D;
	}// java to sm

	public static SimpleMatrix java_to_SimpleMatrix_1D(int[] vector) {
		SimpleMatrix sm1D = new SimpleMatrix(vector.length, 1);
		for (int i = 0; i < vector.length; i++) {
			sm1D.set(i, vector[i]);
		}
		return sm1D;
	}// java to sm

	public static double[] simpleMatrix_to_java_1D_float(SimpleMatrix sm1D) {
		double[] vector = new double[sm1D.numRows()];
		if (sm1D.numCols() > 1) {
			System.out.println("Check SimpleMatrix dimension, is should be [rows = n , cols = 1 ] for 1D");
			return null;
		}
		for (int i = 0; i < sm1D.numRows(); i++) {
			vector[i] = sm1D.get(i);
		}
		return vector;
	}// sm to java

	public static int[] simpleMatrix_to_java_1D_int(SimpleMatrix sm1D) {
		int[] vector = new int[sm1D.numRows()];
		if (sm1D.numCols() > 1) {
			System.out.println("Check SimpleMatrix dimension, is should be [rows = n , cols = 1 ] for 1D");
			return null;
		}
		for (int i = 0; i < sm1D.numRows(); i++) {
			vector[i] = (int) sm1D.get(i);
		}
		return vector;
	}// sm to java

	/////////////////////////   2 D  ///////////////////////////////
	public static SimpleMatrix java_to_SimpleMatrix_2D(double[][] mat2D) {
		SimpleMatrix sm2D = new SimpleMatrix(mat2D.length, mat2D[0].length);
		for (int i = 0; i < mat2D.length; i++) {
			for (int j = 0; j < mat2D[0].length; j++) {
				sm2D.set(i, j, mat2D[i][j]);
			}
		}
		return sm2D;
	}// java to sm

	public static SimpleMatrix java_to_SimpleMatrix_2D(int[][] mat2D) {
		SimpleMatrix sm2D = new SimpleMatrix(mat2D.length, mat2D[0].length);
		for (int i = 0; i < mat2D.length; i++) {
			for (int j = 0; j < mat2D[0].length; j++) {
				sm2D.set(i, j, mat2D[i][j]);
			}
		}
		return sm2D;
	}// java to sm

	public static double[][] simpleMatrix_to_java_2D_float(SimpleMatrix sm2D) {
		double[][] mat = new double[sm2D.numRows()][sm2D.numCols()];
		for (int i = 0; i < sm2D.numRows(); i++) {
			for (int j = 0; j < sm2D.numCols(); j++) {
				mat[i][j] = sm2D.get(i, j);
			}
		}
		return mat;
	}// sm to java

	public int[][] simpleMatrix_to_java_2D_int(SimpleMatrix sm2D) {
		int[][] mat = new int[sm2D.numRows()][sm2D.numCols()];
		for (int i = 0; i < sm2D.numRows(); i++) {
			for (int j = 0; j < sm2D.numCols(); j++) {
				mat[i][j] = (int) sm2D.get(i, j);
			}
		}
		return mat;
	}// sm to java

	public static DenseMatrix64F simpleMatrix_to_Dense(SimpleMatrix sm) {
		return sm.getMatrix();
	}

	public static CommonOps commonOpsObject() {
		//CommonOps comnOps = new CommonOps();
		return new CommonOps();
	}

	public static SimpleMatrix getSimpleMatrixArrayListOfList(ArrayList<ArrayList<Integer>> listOfList) {
		int rows = listOfList.size();
		int cols = (listOfList.get(0)).size();
		int i = 0;
		int j = 0;
		SimpleMatrix sm = new SimpleMatrix(rows, cols);
		for (ArrayList<Integer> rowOfList : listOfList) {
			j = 0;
			for (Integer cellVal : rowOfList) {
				sm.set(i, j, cellVal);
				j++;
			}
			i++;
		}
		return sm;
	}

	public static DenseMatrix64F getDenseMatrix64F_ArrayListOfList(ArrayList<ArrayList<Integer>> listOfList) {
		int rows = listOfList.size();
		int cols = (listOfList.get(0)).size();
		int i = 0;
		int j = 0;
		DenseMatrix64F dense = new DenseMatrix64F(rows, cols);
		for (ArrayList<Integer> rowOfList : listOfList) {
			j = 0;
			for (Integer cellVal : rowOfList) {
				dense.set(i, j, cellVal);
				j++;
			}
			i++;
		}
		return dense;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// 										MATH Operators Linear Algebra
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//return a new DenseMatrix64F parameters unaffected
	public static DenseMatrix64F add(DenseMatrix64F A, DenseMatrix64F B) {
		if (A.numRows == B.numRows && A.numCols == B.numCols) {
			DenseMatrix64F C = new DenseMatrix64F(A.numRows, A.numCols);
			CommonOps.add(A, B, C);
			return C;
		} else {
			System.out.println("Not suitable for add : A: [" + A.numRows + " x " + A.numCols + "] and B: [" + B.numRows + " x " + B.numCols);
			return null;
		}
	}

	public static DenseMatrix64F add(DenseMatrix64F A, double value) {
		DenseMatrix64F B = A.copy();
		CommonOps.add(B, value);
		return B;
	}

	public static DenseMatrix64F add(double value, DenseMatrix64F A) {
		DenseMatrix64F B = A.copy();
		CommonOps.add(B, value);
		return B;
	}

	public static DenseMatrix64F subtract(DenseMatrix64F A, DenseMatrix64F B) {
		if (A.numRows == B.numRows && A.numCols == B.numCols) {
			DenseMatrix64F C = new DenseMatrix64F(A.numRows, A.numCols);
			CommonOps.sub(A, B, C);
			return C;
		} else {
			System.out.println("Not suitable for subtract : A: [" + A.numRows + " x " + A.numCols + "] and B: [" + B.numRows + " x " + B.numCols);
			return null;
		}
	}

	public static DenseMatrix64F subtract(DenseMatrix64F A, double value) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, A.get(i) - value);
		}
		return B;
	}

	public static DenseMatrix64F subtract(double value, DenseMatrix64F A) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, value - A.get(i));
		}
		return B;
	}

	public static DenseMatrix64F dot(DenseMatrix64F A, DenseMatrix64F B) {
		if (A.numCols == B.numRows) {
			DenseMatrix64F C = new DenseMatrix64F(A.numRows, B.numCols);
			CommonOps.mult(A, B, C);
			return C;
		} else {
			System.out.println("Not suitable for dot product: A: [" + A.numRows + " x " + A.numCols + "] and B: [" + B.numRows + " x " + B.numCols+"]");
			return null;
		}
	}

	public static DenseMatrix64F multiply(DenseMatrix64F A, DenseMatrix64F B) {
		if (A.numRows == B.numRows && A.numCols == B.numCols) {
			DenseMatrix64F C = new DenseMatrix64F(A.numRows, A.numCols);
			CommonOps.elementMult(A, B, C);
			return C;
		} else {
			System.out.println("Not suitable for multply : A: [" + A.numRows + " x " + A.numCols + "] and B: [" + B.numRows + " x " + B.numCols+"]");
			return null;
		}
	}

	public static DenseMatrix64F multiply(DenseMatrix64F A, double value) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, A.get(i) * value);
		}
		return B;
	}

	public static DenseMatrix64F multiply(double value, DenseMatrix64F A) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, A.get(i) * value);
		}
		return B;
	}

	public static DenseMatrix64F divide(DenseMatrix64F A, DenseMatrix64F B) {
		if (A.numRows == B.numRows && A.numCols == B.numCols) {
			DenseMatrix64F C = new DenseMatrix64F(A.numRows, A.numCols);
			CommonOps.elementDiv(A, B, C);
			return C;
		} else {
			System.out.println("Not suitable for divide : A: [" + A.numRows + " x " + A.numCols + "] and B: [" + B.numRows + " x " + B.numCols+"]");
			return null;
		}
	}

	public static DenseMatrix64F divide(DenseMatrix64F A, double value) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, A.get(i) / value);
		}
		return B;
	}

	public static DenseMatrix64F divide(double value, DenseMatrix64F A) {
		//DenseMatrix64F B = A.copy();
		//CommonOps.divide(value, B);
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, value / A.get(i));
		}
		return B;
	}

	public static DenseMatrix64F power(DenseMatrix64F A, double value) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, Math.pow(A.get(i), value));
		}
		return B;
	}

	public static DenseMatrix64F sqrt(DenseMatrix64F A) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, Math.sqrt(A.get(i)));
		}
		return B;
	}

	public static DenseMatrix64F abs(DenseMatrix64F A) {
		DenseMatrix64F B = new DenseMatrix64F(A.numRows, A.numCols);
		for (int i = 0; i < A.getNumElements(); i++) {
			B.set(i, Math.abs(A.get(i)));
		}
		return B;
	}

	public static double mean(DenseMatrix64F A) {
		return CommonOps.elementSum(A) / A.getNumElements();
	}

	public static void main(String[] args) {

		int rows = 3;
		int cols = 2;

		SimpleMatrix AS = new SimpleMatrix(rows, cols);
		for (int i = 0; i < AS.getNumElements(); i++) {
			AS.set(i, i);
		}
		SimpleMatrix BS = new SimpleMatrix(rows, cols);
		for (int i = 0; i < AS.getNumElements(); i++) {
			BS.set(i, 9 - i + 1);
		}
		AS.printDimensions();
		//A.print();
		//B.print();

		System.out.println("Testiing Custom Maths");

		DenseMatrix64F A = AS.getMatrix();
		DenseMatrix64F B = BS.getMatrix();

		System.out.println("Dot product test");
		CommonOps.transpose(B);
		//A.print();
		//B.print();
		DenseMatrix64F C = JavaEjml.dot(A, B);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Multiply Matrix test");
		CommonOps.transpose(B);
		//A.print();
		//B.print();
		C = JavaEjml.multiply(A, B);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Multiply numeric test");
		//A.print();
		//B.print();
		C = JavaEjml.multiply(2, A);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Divide Matrix test");
		//A.print();
		//B.print();
		C = JavaEjml.divide(A, B);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Divide numeric test 1: A / value");
		//A.print();
		C = JavaEjml.divide(A, 2);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Divide numeric test 2:  value / A");
		//A.print();
		C = JavaEjml.divide(2, A);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Add Matrix test");
		//A.print();
		//B.print();
		C = JavaEjml.add(A, B);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Add numeric test");
		//A.print();
		C = JavaEjml.add(2, A);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Sub Matrix test");
		//A.print();
		//B.print();
		C = JavaEjml.subtract(B, A);
		//A.print();
		//B.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Sub numeric test 1:  A - value");
		//A.print();
		C = JavaEjml.subtract(A, 2);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Sub numeric test 2:  value - A");
		//A.print();
		C = JavaEjml.subtract(2, A);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Abs  test ");
		//A.print();
		C = JavaEjml.abs(C);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Pow  test");
		//A.print();
		C = JavaEjml.power(A, 2);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Sqrt  test ");
		//A.print();
		C = JavaEjml.sqrt(A);
		//A.print();
		System.out.println("Result:");
		C.print();

		System.out.println("Mean  test "+JavaEjml.mean(A));

		System.out.println("Check Original:");
		A.print();
		B.print();
	}

}