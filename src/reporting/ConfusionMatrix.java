package reporting;

public class ConfusionMatrix {
	int[] truth;
	int[] predicted;
	int catSize;
	private int examples;

	public int[][] confusionMatrix;

	public double error_rate;

	public double[] precision;
	public double[] sensitivity;
	public double[] specificity;

	public int[] tp; // hit rate/ correct prediction rate
	public int[] tn; // correct rejection rate
	public int[] fp; // false alarm, Type I error // rate of incorrectly predicting a cat
	public int[] fn; // with miss, Type II error // rate of incorrectly predicting a cat

	public double accuracy;


	/**
	 *
	 * @param p_truth int[][] target outputs in the form
	 * @param p_predicted int predicted outputs in the form
	 * 100 - class 0
	 * 010 - class 1
	 * 001 - class 2
	 */
	public ConfusionMatrix(int[][] p_truth, int[][] p_predicted) {
		if (p_truth.length == p_predicted.length && p_truth[0].length == p_predicted[0].length) {
			this.examples = p_truth.length;
			this.catSize = p_truth[0].length;
			classCategoriesRetirval(p_truth, p_predicted);
			confusionMatrix();
		} else {
			System.out.println("enter equal shape matrix of truth and predicted outputs");
			System.exit(0);
		}
	}//constructor

	/**
	 *
	 * @param p_truth double[][] target outputs in the form
	 * @param p_predicted int predicted outputs in the form
	 * 1.0 0.0 0.0 - class 0
	 * 0.0 1.0 0.0 - class 1
	 * 0.0 0.0 1.0 - class 2
	 */
	public ConfusionMatrix(double[][] p_truth, double[][] p_predicted) {
		if (p_truth.length == p_predicted.length && p_truth[0].length == p_predicted[0].length) {
			this.examples = p_truth.length;
			this.catSize = p_truth[0].length;
			classPredictionBinarization(p_truth, p_predicted);
			confusionMatrix();
		} else {
			System.out.println("enter equal shape matrix of truth and predicted outputs");
			System.exit(0);
		}
	}//constructor

	/**
	 *
	 * @param p_truth int[] target outputs in the form numeric values
	 * @param p_predicted int predicted outputs in the form
	 * 0 - class 0
	 * 1 - class 1
	 * 2 - class 2
	 * @param p_catSize
	 */
	public ConfusionMatrix(int[] p_truth, int[] p_predicted, int p_catSize) {
		if (p_truth.length == p_predicted.length) {
			this.examples = p_truth.length;
			this.catSize = p_catSize;
			this.truth = p_truth;
			this.predicted = p_predicted;
			confusionMatrix();
		} else {
			System.out.println("enter equal shape matrix of truth and predicted outputs");
			System.exit(0);
		}
	}//constructor

	/**
	 * converting binary pattern of class into single column of numeric values
	 * @param p_truth
	 * @param p_predicted
	 */
	private void classCategoriesRetirval(int[][] p_truth, int[][] p_predicted) {
		this.truth = new int[examples];
		this.predicted = new int[examples];

		int t_cat_ensure, p_cat_ensure, t_cat_val, p_cat_val;

		for (int i = 0; i < examples; i++) {//for all samples
			t_cat_ensure = 0;
			p_cat_ensure = 0;
			t_cat_val = -1;
			p_cat_val = -1;
			//the j-th 1 is the class e.g 1 0 0 is output then j = 0 is set to 1, this is the target and k should be 1
			//e.g 0 1 0 is output then j = 1 is the target and k should be 1
			for (int j = 0; j < catSize; j++) {
				if (p_truth[i][j] == 1) {
					t_cat_ensure++;
					t_cat_val = j;//found target class
				}
				if (p_predicted[i][j] == 1) {
					p_cat_ensure++;
					p_cat_val = j;//found predicted class
				}
			} //go next cat

			//if "t_cat_ensure" NOT EQUAL 1 then there is more than one 1's then the pattern is not a valid target
			if (t_cat_ensure == 1) {
				truth[i] = t_cat_val;
			} else {
				System.out.println("truth " + catSize + " DO NOT Belongs to the any class");
				System.exit(0);
			}
			//if "p_cat_ensure" NOT EQUAL 1 then there is more than one 1's then the pattern is not a valid prediction
			if (p_cat_ensure == 1) {
				predicted[i] = p_cat_val;
			} else {
				System.out.println("truth " + catSize + " DO NOT Belongs to the any class");
				System.exit(0);
			}
			//System.out.println(truth[i] + "  - " + predicted[i]);
		} //go next pat
	}//end process file

	public void classPredictionBinarization(double[][] p_truth, double[][] p_predicted) {
		int[][] y_true = new int[examples][catSize];
		int[][] y_pred = new int[examples][catSize];

		for (int i = 0; i < examples; i++) {
			int max_indx = 0;
			double max_val = p_predicted[i][max_indx];
			for (int j = 0; j < catSize; j++) {
				if (p_predicted[i][j] > max_val) {
					max_indx = j;
					max_val = p_predicted[i][max_indx];
				}
				y_true[i][j] = (int) p_truth[i][j];// just for casting double to int
			}
			y_pred[i][max_indx] = 1;
			//System.out.println(max_val);
		}
		classCategoriesRetirval(y_true, y_pred);
	}

	public void confusionMatrix() {
		this.confusionMatrix = new int[catSize][catSize];// [predicted cats in rows][truth cats in col]

		this.tp = new int[catSize]; // hit rate/ correct prediction rate
		this.tn = new int[catSize]; // correct rejection rate
		this.fp = new int[catSize]; // false alarm, Type I error // rate of incorrectly predicting a cat
		this.fn = new int[catSize]; // with miss, Type II error // rate of incorrectly predicting a cat
		/* Table: Truth - Prediction Orientation Confusion Matrix
		// https://en.wikipedia.org/wiki/Confusion_matrix
		//-----------------------------------------------------------------------
		//						|				Predicted class      		  |
		//-----------------------------------------------------------------------
		//			 |		    |          Cat	         |     Non-cat		  |
		// Actual    |----------------------------------------------------------
		// class	 | Cat	    | 5 True Positives	     |   False Negatives  |
		// (truth)   | Non-cat	| 3 False Positives      |   True Negatives   |
		//-----------------------------------------------------------------------
		precision or positive predictive value (PPV) = TP/(TP+FP)
		sensitivity, recall, hit rate, or true positive rate (TPR) = TP/P = TP/(TP+FN)
		specificity, selectivity or true negative rate (TNR) = TN/N = = TN/(TN+FP)
		For multi class
		tpA	tp_i = m_ii
		tpB	tp_i = m_ii
		tpC	tp_i = m_ii

		tnA	sum_mat - tp_i
		tnB	sum_mat - tp_i
		tnC	sum_mat - tp_i

		(if "truth-pred" mat orient of Table)	(if "pred-truth" mat orient of Table)
		fpA	sumC_i-tp_i							sumR_i-tp_i
		fpB	sumC_i-tp_i							sumR_i-tp_i
		fpC	sumC_i-tp_i							sumR_i-tp_i

		fnA	sumR_i-tp_i							sumC_i-tp_i
		fnB	sumR_i-tp_i							sumC_i-tp_i
		fnC	sumR_i-tp_i							sumC_i-tp_i
		 */
		this.precision = new double[catSize];
		this.sensitivity = new double[catSize];
		this.specificity = new double[catSize];

		//finding values for the confusion matrix
		for (int i = 0; i < examples; i++) {//for all samples
			if (truth[i] < catSize) {
				if (predicted[i] < catSize) {
					//[truth cats in col][predicted cats in rows]
					confusionMatrix[truth[i]][predicted[i]] = confusionMatrix[truth[i]][predicted[i]] + 1;

					//if [predicted cats in rows][truth cats in col] as shown in wikipedia use this following
					//confusionMatrix[predicted[i]][truth[i]] = confusionMatrix[predicted[i]][truth[i]] + 1;
				} else {
					System.out.println("predicted class " + predicted[i] + "does not exist ");
					System.exit(0);
				}
			} else {
				System.out.println("truth class " + truth[i] + "does not exist ");
				System.exit(0);
			}
		} // for all samples

		//System.out.println("\n\nClassification Statistics :");
		int sum_mat = 0;
		int sum_dig = 0;

		for (int i = 0; i < catSize; i++) {
			tp[i] = confusionMatrix[i][i]; //actual animals were correctly classified as what they are
			for (int j = 0; j < catSize; j++) {
				if (i != j) {
					//cat was incorrectly labelled as all the remaining animals
					fn[i] = fn[i] + confusionMatrix[i][j]; // eqv. sumR_i -- mat_ii
					//all the remaining animals that were incorrectly labelled as cat
					fp[i] = fp[i] + confusionMatrix[j][i]; // eqv. sumC_i -- mat_ii
				}
				//System.out.print(" " + confusionMat[i][j]);
				sum_mat = sum_mat + confusionMatrix[i][j];
			} //end predicted class cols
			sum_dig = sum_dig + tp[i];
		} //end truth class rows
		for (int i = 0; i < catSize; i++) {
			tn[i] = sum_mat - (tp[i] + fn[i] + fp[i]); // all the remaining animals correctly classified as what they actually are

			//System.out.format("class " + i);
			precision[i] = ((tp[i] + fp[i]) == 0.0) ? 0.0 : (double) (tp[i] / ((double) (tp[i] + fp[i])));//(PPV) = TP/(TP+FP)
			sensitivity[i] = ((tp[i] + fn[i]) == 0.0) ? 0.0 : (double) (tp[i] / ((double) (tp[i] + fn[i])));//(TPR) = TP/(TP+FN)
			specificity[i] = ((tp[i] + fn[i]) == 0.0) ? 0.0 : (double) (tn[i] / ((double) (tn[i] + fp[i])));//(TNR) = TN/(TN+FP)
		} //for
		this.accuracy = (sum_dig) / ((double) examples);
		this.error_rate = 1.0 - accuracy;
	}//end of confusion matrix computation

	public static void main(String[] args) {
		//All tested OK
		int[] t = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
		int[] p = { 0, 1, 0, 1, 2, 0, 0, 2, 2, 1, 2, 1, 2, 1, 2 };

		//int[][] t = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 },
		//		{ 0, 0, 1 }, { 0, 0, 1 } };
		//int[][] p = { { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 0 },
		//		{ 0, 1, 0 }, { 0, 0, 1 } };

		//double[][] td = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 },
		//		{ 0, 0, 1 }, { 0, 0, 1 } };
		//double[][] pd = { { 0.8, 0.01, 0 }, { 0, 0.81, 0 }, { 0.8, 0.01, 0 }, { 0, 0.81, 0 }, { 0, 0.01, 0.8 }, { 0.8, 0.01, 0 }, { 0.8, 0.01, 0 }, { 0, 0.81, 0 }, { 0, 0.01, 0.8 }, { 0, 0.81, 0 },
		//		{ 0, 0.01, 0.8 }, { 0, 0.81, 0 }, { 0, 0.81, 0 }, { 0, 0.81, 0 }, { 0, 0.01, 0.8 } };

		//ConfusionMatrix cm = new ConfusionMatrix(td, pd);
		//ConfusionMatrix cm = new ConfusionMatrix(t, pd);

		ConfusionMatrix cm = new ConfusionMatrix(t, p, 3);
		System.out.println(cm.accuracy);
		int[][] cmM = cm.confusionMatrix;
		for (int i = 0; i < 3; i++) {
			/*System.out.print(" " + cm.tp[i]);
			System.out.print(" " + cm.fn[i]);
			System.out.print(" " + cm.fp[i]);
			System.out.print(" " + cm.tn[i]);
			System.out.print(" " + cm.tn[i]);*/
			double sensitivity = ((cm.tp[i] + cm.fn[i]) == 0.0) ? 0.0 : (double) (cm.tp[i] / ((double) (cm.tp[i] + cm.fn[i])));//(TPR) = TP/(TP+FN)
			double specificity = ((cm.tp[i] + cm.fn[i]) == 0.0) ? 0.0 : (double) (cm.tn[i] / ((double) (cm.tn[i] + cm.fp[i])));//(TNR) = TN/(TN+FP)
			double precision = ((cm.tp[i] + cm.fp[i]) == 0.0) ? 0.0 : (double) (cm.tp[i] / ((double) (cm.tp[i] + cm.fp[i])));//(PPV) = TP/(TP+FP)
			System.out.print(" " + precision);
			System.out.print(" " + sensitivity);
			System.out.print(" " + specificity);
			/*for (int j = 0; j < 3; j++) {
				System.out.print("\t" + cmM[i][j]);
			}*/
			System.out.println();
		}
	}
}
