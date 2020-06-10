package optimization.gd;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.ejml.data.DenseMatrix64F;

import experimentSettings.ExperimentParameters;
import javaEjml.JavaEjml;
import neuralTree.EvaluateTree;
import neuralTree.NeuralTree;
import optimization.CostFunction;
import optimization.Individual;
import reporting.SaveAlgorithmRun;


public class GD {
	EvaluateTree mEvaluateTree;// Tree evaluation parameters
	ExperimentParameters mExpParams;//  set of parameters
	NeuralTree mTree;//  set of parameters

	private int p_target_attr_count;

	private int mMaxIt; // Maximum Number of Iterations

	private DenseMatrix64F curTheta;
	private DenseMatrix64F vTheta;
	private DenseMatrix64F mTheta;

	private DenseMatrix64F bestTreeTheta;
	ArrayList<Object> bestThetaCostAll;
	double bestThetaError;
	//double bestThetaErrorTst;

	private ArrayList<double[]> performance_record_stocstic;
	private ArrayList<Object> performance_record_batch;

	CostFunction costFunction;
	private String costOf;
	private String mGDMode;
	private String mGDAlgorithm;


	public GD(EvaluateTree ev, ExperimentParameters exp_params, NeuralTree nTree, String algoName) {
		this.mEvaluateTree = ev;
		this.mExpParams = exp_params;
		this.mTree = nTree;

		this.p_target_attr_count = exp_params.n_max_target_attr;

		// GD parameters
		this.mMaxIt = mExpParams.n_param_opt_max_itr; // Maximum Number of Iterations
		this.mGDMode = mExpParams.n_gd_eval_mode;
		//this.mGDAlgorithm = mExpParams.n_algo_param;
		this.mGDAlgorithm = algoName;

		this.bestTreeTheta = null;

		costFunction = new CostFunction(mEvaluateTree, 2);
		this.costOf = exp_params.n_check_epoch_set;

		this.performance_record_batch = new ArrayList<Object>();
		this.performance_record_stocstic = new ArrayList<double[]>();

	}

	public Individual start(File directory, String trial) {
		curTheta = mTree.getWeightBiasDense(p_target_attr_count);
		bestTreeTheta = curTheta.copy();/// current theta is the best theata

		//mTree.displayTree();
		mEvaluateTree.set_dataset_to_evaluate("train");// take training set
		double[][] X = mEvaluateTree.m_data_input_TO_tree;
		double[][] Y = mEvaluateTree.m_data_target_TO_tree;


		long start = System.currentTimeMillis();
		bestThetaCostAll = costFunction(costOf);
		bestThetaError = costFunction.costOnlyObjectives(bestThetaCostAll)[0];
		long finish = System.currentTimeMillis();
		double timeElapsed = (finish - start) / 1000.0;
		System.out.println("Algo "+mGDAlgorithm + " best tree theta: " + Arrays.toString(costFunction.costOnlyObjectives(bestThetaCostAll))+" compute time: "+timeElapsed+" sec.");
		performance_record_batch.add(bestThetaCostAll);
		double startError = bestThetaError;

		ArrayList<Integer> randIndex = new ArrayList<Integer>();
		if (mGDMode.equalsIgnoreCase("stochastic") || mGDMode.equalsIgnoreCase("mini_batch")) {
			double[] record_stoc = new double[X.length];
			for (int i = 0; i < X.length; i++) {
				randIndex.add(i);
				record_stoc[i] = startError;
			}
			performance_record_stocstic.add(record_stoc);
		} // update record stochastic

		int iters = 0; //iteration counter
		int change_counter = 0;
		double previous_step_size = Double.POSITIVE_INFINITY; //
		vTheta = new DenseMatrix64F(1, curTheta.numCols); //  Initialise to zeros
		mTheta = new DenseMatrix64F(1, curTheta.numCols); //  Initialise to zeros

		while (previous_step_size > mExpParams.n_gd_precision && iters < mMaxIt) {
			//while (iters < mMaxIt) {
			//Preserver current weights and bias in prev_n-1
			DenseMatrix64F prevTheta = curTheta.copy();
			if (mGDMode.equalsIgnoreCase("stochastic")) {
				double[] record_stocasticity = new double[X.length];
				//int num_update = 0;
				Collections.shuffle(randIndex);// Shuffle
				//System.out.println("Shuffled:"+randIndex);
				double[][] rX = new double[1][X[0].length];// fetch one example input.
				double[][] rY = new double[1][Y[0].length];// fetch one example output
				for (int i = 0; i < X.length; i++) {
					prevTheta = curTheta.copy();
					//num_update += 1;
					rX[0] = X[randIndex.get(i)];
					rY[0] = Y[randIndex.get(i)];
					//System.out.println(Arrays.toString(rX[0])+" "+Arrays.toString(rY[0]));
					/*try {
						Thread.sleep(1 * 1000);
					} catch (InterruptedException ie) {
						Thread.currentThread().interrupt();
					}
					System.out.println("for "+i); */
					update_theta(rX, rY);//curTheta, vTheta, mTheta = update_theta(x, y, curTheta, vTheta, mTheta, num_update, 'stochastic')
					// Updating weights and bias tree by evaluating the tree
					ArrayList<Object> currThetaErrorAll = costFunction(costOf);
					double currThetaError = costFunction.costOnlyObjectives(currThetaErrorAll)[0];
					//System.out.println(i+" : "+currThetaError);
					//Checking best theta (tree parameter)
					if (currThetaError < bestThetaError) {
						change_counter += 1;
						bestThetaCostAll = currThetaErrorAll;
						bestThetaError = currThetaError;
						bestTreeTheta = curTheta.copy();//copy best parameter
						// Open this to save every time a new best is found.
						System.out.println("New best *"+costOf+"* : "+ bestThetaError+" when: "+java.time.LocalDateTime.now());
						saveBest(bestTreeTheta, directory, trial);
						change_counter += 1;
					}
					record_stocasticity[i] = currThetaError;
					//previous_step_size = JavaEjml.mean(JavaEjml.abs(JavaEjml.subtract(curTheta, prevTheta)));
					/*if (previous_step_size < mExpParams.n_gd_precision) {
						System.out.println("Weight change speed very slow: " + previous_step_size);
						break;//
					} //break for loop */
				} //end weight update for all examples
				//System.exit(0);
				performance_record_stocstic.add(record_stocasticity);
				performance_record_batch.add(bestThetaCostAll);
			} if (mGDMode.equalsIgnoreCase("mini_batch")) {
				ArrayList<Double> record_stocasticity = new ArrayList<Double>();
				//int num_update = 0;
				int examples_length =  X.length;
				int bacth_reducer = examples_length;
				int batch_size = mExpParams.n_batch_size;
				int index_counter = 0;
				int batch_current = -1;

				Collections.shuffle(randIndex);// Shuffle
				//System.out.println("Shuffled:"+randIndex.size());
				//System.out.println("Mini - batch start: "+index_counter);
				while (bacth_reducer > 0){
					//System.out.print(bacth_reducer +" : when: "+java.time.LocalDateTime.now());
					if(bacth_reducer < batch_size) {
						batch_current = bacth_reducer;
					}else {
						batch_current = batch_size;
					}
					//System.out.print(batch_current +" : ");
					double[][] bX = new double[batch_current][X[0].length];// fetch one example input.
					double[][] bY = new double[batch_current][Y[0].length];// fetch one example output
					for(int i =0; i < batch_current;i++) {
						int index = randIndex.get(index_counter + i);
						bX[i] = X[index];
						bY[i] = Y[index];
					}//end all batch taken up update for all examples
					//System.out.println();
					bacth_reducer = bacth_reducer - batch_size;
					index_counter = index_counter + batch_current;

					// Update MINI BATCH
					prevTheta = curTheta.copy();
					//num_update += 1;
					update_theta(bX, bY);//curTheta, vTheta, mTheta = update_theta(x, y, curTheta, vTheta, mTheta, num_update, 'stochastic')
					// Updating weights and bias tree by evaluating the tree
					ArrayList<Object> currThetaErrorAll = costFunction(costOf);
					double currThetaError = costFunction.costOnlyObjectives(currThetaErrorAll)[0];
					//System.out.println(i+" : "+currThetaError);
					//Checking best theta (tree parameter)
					if (currThetaError < bestThetaError) {
						change_counter += 1;
						bestThetaCostAll = currThetaErrorAll;
						bestThetaError = currThetaError;
						bestTreeTheta = curTheta.copy();//copy best parameter
						// Open this to save every time a new best is found.
						System.out.println("New best *"+costOf+"* : "+ bestThetaError+" when: "+java.time.LocalDateTime.now());
						saveBest(bestTreeTheta, directory, trial);
						change_counter += 1;
					}
					record_stocasticity.add(currThetaError);
					//previous_step_size = JavaEjml.mean(JavaEjml.abs(JavaEjml.subtract(curTheta, prevTheta)));
					/*if (previous_step_size < mExpParams.n_gd_precision) {
						System.out.println("Weight change speed very slow: " + previous_step_size);
						break;//
					} //break for loop */
				}// end of all batch
				//System.out.println("Mini - batch end: "+index_counter);
				//System.exit(0);
				performance_record_stocstic.add(record_stocasticity.stream().mapToDouble(Double::doubleValue).toArray());
				performance_record_batch.add(bestThetaCostAll);
			} else {//batch training
				//int num_update = 1;
				update_theta(X, Y); //curTheta, vTheta, mTheta = update_theta(X, Y, curTheta, vTheta, mTheta, num_update)
				// Updating weights and bias tree by evaluating the tree
				ArrayList<Object> currThetaErrorAll = costFunction(costOf);
				double currThetaError = costFunction.costOnlyObjectives(currThetaErrorAll)[0];
				//Checking best theta (tree parameter)
				if (currThetaError < bestThetaError) {
					System.out.println("New best: " + bestThetaError);
					bestThetaCostAll = currThetaErrorAll;
					bestThetaError = currThetaError;
					bestTreeTheta = curTheta.copy();//copy best parameter
					System.out.println("New best * "+costOf+"* : " + bestThetaError+" when: "+java.time.LocalDateTime.now());
					saveBest(bestTreeTheta, directory, trial);
					change_counter += 1;
				}
				previous_step_size = JavaEjml.mean(JavaEjml.abs(JavaEjml.subtract(curTheta, prevTheta)));
				performance_record_batch.add(bestThetaCostAll);
			} //end modes of GD */
			System.out.printf("%d : best %s : %.4f  when: %s \n",iters, costOf, bestThetaError, java.time.LocalDateTime.now().toString());
			//saveBest(bestTreeTheta, directory, trial+"_"+iters);
			iters++;
		} //end GD training while loop
		System.out.printf("GD: %s best %s @start: %.4f  @end: %.4f  cahnge: %d \n", mGDAlgorithm, costOf, startError, bestThetaError, change_counter);
		SaveAlgorithmRun.saveGpIteration(performance_record_batch, directory, trial, mGDAlgorithm, mExpParams.n_problem_type, mExpParams.n_max_target_attr);
		if (mGDMode.equalsIgnoreCase("stochastic")) {
			SaveAlgorithmRun.saveSGDIteration(performance_record_stocstic, directory, trial, mGDAlgorithm);
		}
		saveBest(bestTreeTheta, directory, trial);
		Individual tree = new Individual();
		mTree.setWeightBaisDense(bestTreeTheta, p_target_attr_count);
		tree.mTree = mTree;
		tree.mCostAll = bestThetaCostAll;
		tree.mCost = costFunction.costOnlyObjectives(bestThetaCostAll);
		return tree;
	}//GD ends

	private void saveBest(DenseMatrix64F bestTreeTheta2, File directory, String trial) {
		mTree.setWeightBaisDense(bestTreeTheta2, p_target_attr_count);
		mTree.saveTreeModel(mTree, directory, trial+"_optPRM_"+mGDAlgorithm, mExpParams.n_max_target_attr, mExpParams.n_data_input_names, mExpParams.n_data_target_names, false);
		//System.out.println(); (trial + "_optPRM_" + algoName)
	}

	private ArrayList<Object> costFunction(String costOfset) {
		return costFunction.costAll(mTree, costOfset);
	}

	private void update_theta(double[][] X1, double[][] Y1) {
		//rmsprop(X1, Y1);

		// / * switch case for check computation every time
		switch (mGDAlgorithm) {
		case "rmsprop":
			//System.out.println(mGDAlgorithm);
			rmsprop(X1, Y1);
			break;

		case "adam":
			// curTheta, vTheta = momentum_gd(X, Y, curTheta, vTheta, evl_type)
			adam(X1, Y1);
			break;

		case "nesterov_accelerated_gd":
			// curTheta, vTheta = nesterov_accelerated_gd(X, Y, curTheta, vTheta, evl_type)
			nesterov_accelerated_gd(X1, Y1);
			break;

		case "momentum_gd":
			// curTheta, vTheta = adagrad(X, Y, curTheta, vTheta, evl_type)
			momentum_gd(X1, Y1);
			break;

		case "adagrad":
			// curTheta, vTheta = rmsprop(X, Y, curTheta, vTheta, evl_type)
			adagrad(X1, Y1);
			break;

		case "gd":
			//curTheta, vTheta, mTheta = adam(X, Y, curTheta, vTheta, mTheta, num_update, evl_type)
			gradient_descent(X1, Y1);
			break;

		default:
			System.out.println("no algorithmn on this name");
		} // */
		//Update tree theta
		mTree.setWeightBaisDense(curTheta, p_target_attr_count);
	}//end updated theata

	private DenseMatrix64F evaluate_gradient(double[][] X1, double[][] Y1) {
		if (mGDMode.equalsIgnoreCase("stochastic")) {
			return mTree.getGradientDense(X1[0], Y1[0], p_target_attr_count);
		} else {//for batch mode compute average of all examples
			DenseMatrix64F gradient = new DenseMatrix64F(curTheta.numRows, curTheta.numCols);
			for (int i = 0; i < X1.length; i++) { //X1.length should be 1 for stochastic gradient
				gradient = JavaEjml.add(gradient, mTree.getGradientDense(X1[i], Y1[i], p_target_attr_count));
			} //gradient.print();	System.out.println(X1.length);
			// return the average of gradient for the batch
			return JavaEjml.divide(gradient, X1.length);
		}
	}

	private void gradient_descent(double[][] X1, double[][] Y1) {
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1);
		DenseMatrix64F curThetaChange = JavaEjml.multiply(gradient, mExpParams.n_gd_eta);
		curTheta = JavaEjml.subtract(curTheta, curThetaChange); //JavaEjml.add(curTheta, curThetaChange)
		//curTheta = JavaEjml.add(curTheta, curThetaChange);
	}

	private void momentum_gd(double[][] X1, double[][] Y1) {
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1);
		// Computing components of momentum GD
		vTheta = JavaEjml.multiply(mExpParams.n_gd_gamma, vTheta); // gamma * v(t-1)
		DenseMatrix64F eta_X_grad = JavaEjml.multiply(gradient, mExpParams.n_gd_eta); // eta * grad
		// v(t) = gamma * v(t-1) + eta * grad
		vTheta = JavaEjml.add(vTheta, eta_X_grad);
		// theta = theta - v(t)
		curTheta = JavaEjml.subtract(curTheta, vTheta);
		//curTheta = JavaEjml.add(curTheta, vTheta)
	}

	private void nesterov_accelerated_gd(double[][] X1, double[][] Y1) {
		// Computing components of NGA GD
		vTheta = JavaEjml.multiply(mExpParams.n_gd_gamma, vTheta); // gamma * v(t-1)
		DenseMatrix64F curTheta_minus_gama_vTheta = JavaEjml.subtract(curTheta, vTheta); //  (theta - gamma * v(t-1))
		// set (theta - gamma * v(t-1)) as the weights of the tree
		mTree.setWeightBaisDense(curTheta_minus_gama_vTheta, mExpParams.n_max_target_attr);
		// take the average of gradient for the batch
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1);

		DenseMatrix64F eta_X_grad = JavaEjml.multiply(mExpParams.n_gd_eta, gradient); // eta * grad
		// v(t) = gamma * v(t-1) + eta * grad_on(theta - gamma * v(t-1))
		vTheta = JavaEjml.add(vTheta, eta_X_grad);
		// theta = theta - v(t)
		curTheta = JavaEjml.subtract(curTheta, vTheta);
		//curTheta = JavaEjml.add(curTheta, vTheta);
	}

	private void adagrad(double[][] X1, double[][] Y1) {
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1); // g(t)
		// Computing components of adagrad
		DenseMatrix64F grad_square = JavaEjml.power(gradient, 2); // g(t)^2
		vTheta = JavaEjml.add(vTheta, grad_square); //  v(t) = v(t-1) + g(t)^2
		DenseMatrix64F vTheta_plus_eps = JavaEjml.add(vTheta, mExpParams.n_gd_eps); // v(t) + eps
		DenseMatrix64F sqrt_of_vTheta_plus_eps = JavaEjml.sqrt(vTheta_plus_eps); //  squrt(v(t) + eps)
		DenseMatrix64F eta_by_sqrt_vTheta_plus_eps = JavaEjml.divide(mExpParams.n_gd_eta, sqrt_of_vTheta_plus_eps); // (eta / squrt(v(t) + eps))
		DenseMatrix64F eta_by_sqrt_vTheta_plus_eps_x_grad = JavaEjml.multiply(eta_by_sqrt_vTheta_plus_eps, gradient); // (eta / squrt(v(t) + eps)) * g(t)
		// theta(t) = theta(t-1) -  (eta / squrt(v(t)) + eps) * g(t)
		curTheta = JavaEjml.subtract(curTheta, eta_by_sqrt_vTheta_plus_eps_x_grad);
	}

	private void rmsprop(double[][] X1, double[][] Y1) {
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1); // g(t)
		// Computing components of RMSprop
		DenseMatrix64F grad_square = JavaEjml.power(gradient, 2); // g(t)^2
		DenseMatrix64F beta_x_vTheta = JavaEjml.multiply(vTheta, mExpParams.n_gd_beta); //  beta * v(t-1)
		DenseMatrix64F one_minus_beta_x_grad_square = JavaEjml.multiply((1.0 - mExpParams.n_gd_beta), grad_square); // (1.0 - beta) * g(t)^2
		vTheta = JavaEjml.add(beta_x_vTheta, one_minus_beta_x_grad_square);//  v(t) = beta * v(t-1) + (1.0 - beta) *  g(t)^2

		DenseMatrix64F vTheta_plus_eps = JavaEjml.add(vTheta, mExpParams.n_gd_eps); // vTheata + eps
		DenseMatrix64F sqrt_of_vTheta_plus_eps = JavaEjml.sqrt(vTheta_plus_eps); //  squrt(vTheta + eps)
		DenseMatrix64F eta_by_sqrt_vTheta_plus_eps = JavaEjml.divide(mExpParams.n_gd_eta, sqrt_of_vTheta_plus_eps); // (eta / squrt(v(t) + eps))
		DenseMatrix64F eta_by_sqrt_vTheta_plus_eps_x_grad = JavaEjml.multiply(eta_by_sqrt_vTheta_plus_eps, gradient); // (eta / squrt(v(t) + eps)) * g(t)
		// theta = theta -  (eta / squrt(v(t)) + eps) * g(t)
		curTheta = JavaEjml.subtract(curTheta, eta_by_sqrt_vTheta_plus_eps_x_grad);
	}

	private void adam(double[][] X1, double[][] Y1) {
		int num_update = 1;
		DenseMatrix64F gradient = evaluate_gradient(X1, Y1); // g(t)
		// Computing components of Adam
		DenseMatrix64F grad_square = JavaEjml.power(gradient, 2); // g(t)^2
		double beta1 = mExpParams.n_gd_beta1;
		double beta2 = mExpParams.n_gd_beta2;

		DenseMatrix64F beta1_x_mTheta = JavaEjml.multiply(mTheta, beta1); //  beta1 * m(t-1)
		DenseMatrix64F one_minus_beta1_grad = JavaEjml.multiply((1.0 - beta1), gradient); //  (1.0 - beta1) * g(t)
		mTheta = JavaEjml.add(beta1_x_mTheta, one_minus_beta1_grad); // m(t) =  beta1 * m(t) + (1.0 - beta1) * g(t)

		DenseMatrix64F beta2_x_vTheta = JavaEjml.multiply(vTheta, beta2); //  beta2 * v(t-1)
		DenseMatrix64F one_minus_beta2_grad_square = JavaEjml.multiply((1.0 - beta2), grad_square); //  (1.0 - beta2) * g(t)^2
		vTheta = JavaEjml.add(beta2_x_vTheta, one_minus_beta2_grad_square); // v(t) =  beta2 * v(t) + (1.0 - beta2) * g(t)^2

		double one_minus_beta1_pow_num_update = 1.0 - Math.pow(beta1, num_update);
		DenseMatrix64F m_theta_hat = JavaEjml.divide(mTheta, one_minus_beta1_pow_num_update);

		double one_minus_beta2_pow_num_update = 1.0 - Math.pow(beta2, num_update);
		DenseMatrix64F v_theta_hat = JavaEjml.divide(vTheta, one_minus_beta2_pow_num_update);

		DenseMatrix64F vTheta_hat_plus_eps = JavaEjml.add(v_theta_hat, mExpParams.n_gd_eps); // v(t) + eps
		DenseMatrix64F sqrt_of_vTheta_hat_plus_eps = JavaEjml.sqrt(vTheta_hat_plus_eps); //  squrt(v(t) + eps)
		DenseMatrix64F eta_by_sqrt_vTheta_hat_plus_eps = JavaEjml.divide(mExpParams.n_gd_eta, sqrt_of_vTheta_hat_plus_eps);// (eta / squrt(v_hat(t) + eps))
		DenseMatrix64F eta_by_sqrt_vTheta_hat_plus_eps_x_m_theta_hat = JavaEjml.multiply(eta_by_sqrt_vTheta_hat_plus_eps, m_theta_hat); // // (eta / squrt(v_hat(t) + eps)) * m_hat(t)
		curTheta = JavaEjml.subtract(curTheta, eta_by_sqrt_vTheta_hat_plus_eps_x_m_theta_hat);
	}

}
