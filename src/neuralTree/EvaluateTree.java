package neuralTree;

import java.util.ArrayList;
import java.util.stream.IntStream;

import reporting.ConfusionMatrix;
import reporting.RegressionStat;

public class EvaluateTree {

	String m_y_type;

	double[][] m_training_inputs;
	double[][] m_training_targets;

	double[][] m_test_inputs;
	double[][] m_test_targets;

	double[][] m_val_inputs;
	double[][] m_val_targets;

	boolean tree_inputs_set = false;
	// This is temporary object holders:
	public double[][] m_data_input_TO_tree; // for the input set to be used by the tree to get fitness
	public double[][] m_data_target_TO_tree; // for the target set to be used by the tree to get fitness
	double[][] m_data_prediction_OF_tree; // for the prediction set to be used by the tree to get fitness

	private double[] mTargetMin;
	private double[] mTargetMax;

	private double[] m_scale;

	public EvaluateTree(double[][] m_training_inputs, double[][] m_training_targets, double[][] m_test_inputs, double[][] m_test_targets, double[][] m_val_inputs, double[][] m_val_targets,
			double[] m_targetMin, double[] m_targetMax, double[] n_scale, String problem_type) {

		this.m_y_type = problem_type;
		this.m_training_inputs = m_training_inputs;
		this.m_training_targets = m_training_targets;

		this.m_test_inputs = m_test_inputs;
		this.m_test_targets = m_test_targets;

		this.m_val_inputs = m_val_inputs;
		this.m_val_targets = m_val_targets;

		this.mTargetMin = m_targetMin;
		this.mTargetMax = m_targetMax;
		this.m_scale = n_scale;
	}

	public void set_dataset_to_evaluate(String p_data_to_evaluate) {
		this.m_data_input_TO_tree = null;
		this.m_data_target_TO_tree = null;
		tree_inputs_set = false;

		if (p_data_to_evaluate.equalsIgnoreCase("train")) {
			//System.out.println("Datasets loaded for TRAINING....");
			this.m_data_input_TO_tree = this.m_training_inputs;//train
			this.m_data_target_TO_tree = this.m_training_targets;
			tree_inputs_set = true;
		}
		if (p_data_to_evaluate.equalsIgnoreCase("test")) {
			//System.out.println("Datasets loaded for TESTING....");
			this.m_data_input_TO_tree = this.m_test_inputs;//test
			this.m_data_target_TO_tree = this.m_test_targets;
			tree_inputs_set = true;
		}
		if (p_data_to_evaluate.equalsIgnoreCase("val")) {
			//System.out.println("Datasets loaded for VALIDATING....");
			this.m_data_input_TO_tree = this.m_val_inputs;//val
			this.m_data_target_TO_tree = this.m_val_targets;
			tree_inputs_set = true;
		}
	}


	/**
	 * GPU implementation:  advantages only for large training examples
	 * Evaluate tree fitness for a input detests return the predicted output
	 * takes a tree object to evaluate its fitness (predicted outputs)
	 * @param p_treeToEvalute NeuralTree
	 * @return a(an) matrix/array of tree prediction  rows x column
	 */
	/*
	public double[][] getTreePredictedOutputsGPU(NeuralTree p_treeToEvalute) {
		if (!tree_inputs_set) {
			System.out.println("\nSelect a type of set evaluate tree");
			System.exit(0);
		}
		int m_target_attr_count = m_data_target_TO_tree[0].length;
		int n_examples = m_data_target_TO_tree.length;
		m_data_prediction_OF_tree = new double[m_data_input_TO_tree.length][m_target_attr_count];

		Kernel kernel = new Kernel() {
			@Override
			public void run() {
				int idx = getGlobalId();
				m_data_prediction_OF_tree[idx] = p_treeToEvalute.getOutput(m_data_input_TO_tree[idx], m_target_attr_count);
			}
		};
		Range range = Range.create(n_examples);
		System.out.println("Examples G:"+range);
		kernel.execute(range);

		// return the prediction - this may or many not be used by the a function
		return m_data_prediction_OF_tree;
	} */

	/**
	 * Parallel Stream  -  Use this for large training examples only
	 * Evaluate tree fitness for a input detests return the predicted output
	 * takes a tree object to evaluate its fitness (predicted outputs)
	 * @param p_treeToEvalute NeuralTree
	 * @return a(an) matrix/array of tree prediction  rows x column
	 */
	public double[][] getTreePredictedOutputsParallel(NeuralTree p_treeToEvalute) {
		if (!tree_inputs_set) {
			System.out.println("\nSelect a type of set evaluate tree");
			System.exit(0);
		}
		int m_target_attr_count = m_data_target_TO_tree[0].length;
		int n_examples = m_data_target_TO_tree.length;
		m_data_prediction_OF_tree = new double[m_data_input_TO_tree.length][m_target_attr_count];
		IntStream stream = IntStream.range(0, n_examples);

		//System.out.println("Examples P:"+n_examples);

		stream.parallel().forEach(idx -> {
			m_data_prediction_OF_tree[idx] = p_treeToEvalute.getOutput(m_data_input_TO_tree[idx], m_target_attr_count);
		});
		// return the prediction - this may or many not be used by the a function
		return m_data_prediction_OF_tree;
	}

	/**
	 * Evaluate tree fitness for a input detests return the predicted output
	 * takes a tree object to evaluate its fitness (predicted outputs)
	 * @param p_treeToEvalute NeuralTree
	 * @return a(an) matrix/array of tree prediction  rows x column
	 */
	public double[][] getTreePredictedOutputs(NeuralTree p_treeToEvalute) {
		if (!tree_inputs_set) {
			System.out.println("\nSelect a type of set evaluate tree");
			System.exit(0);
		}
		int m_target_attr_count = m_data_target_TO_tree[0].length;
		int n_examples = m_data_target_TO_tree.length;
		m_data_prediction_OF_tree = new double[m_data_input_TO_tree.length][m_target_attr_count];
		for (int i = 0; i < n_examples; i++) {
			m_data_prediction_OF_tree[i] = p_treeToEvalute.getOutput(m_data_input_TO_tree[i], m_target_attr_count);
		}
		// return the prediction - this may or many not be used by the a function
		return m_data_prediction_OF_tree;
	}

	/**
	 * Evaluate fitness score of the tree
	 * @return ArrayList<Object>
	 * For class
	 * error_rate, precision, sensitivity cm.specificity , tp, fp, fn,tn
	 * For Regression
	 * mse, corr (r), r2
	 */
	public ArrayList<Object> getTreeFitness() {
		return compareTrueAndPred();
	}

	private ArrayList<Object> compareTrueAndPred() {
		ArrayList<Object> performance = new ArrayList<Object>();
		if (this.m_y_type.equalsIgnoreCase("Classification")) {
			//If problem is classification
			// return error rate
			ConfusionMatrix cm = new ConfusionMatrix(m_data_target_TO_tree, m_data_prediction_OF_tree);
			performance.add(cm.error_rate);// error (double)
			performance.add(cm.precision);//PPR (double[])
			performance.add(cm.sensitivity);//recall (double[])
			performance.add(cm.specificity);//TNR (double[])
			performance.add(cm.tp);//tp (int[])
			performance.add(cm.fp);//fp (int[])
			performance.add(cm.fn);//fn (int[])
			performance.add(cm.tn);//tn (int[])
		} else {
			RegressionStat reg = new RegressionStat(m_data_target_TO_tree, m_data_prediction_OF_tree, mTargetMin, mTargetMax, m_scale);
			performance.add(reg.mse);//mse
			performance.add(reg.rxy);//r Correlation Coefficient
			performance.add(reg.r2);// r2 - Coefficient of determination: Nash–Sutcliffe model efficiency coefficient
		}
		return performance;
	}

}
