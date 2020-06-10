package optimization;

import java.io.File;

import experimentSettings.ExperimentParameters;
import neuralTree.EvaluateTree;
import neuralTree.NeuralTree;
import optimization.gd.GD;


public class ParamterOptimization {
	EvaluateTree mEvaluateTree;// Tree evaluation parameters
	ExperimentParameters mParams;//  set of parameters
	NeuralTree mTree;//  set of parameters

	public ParamterOptimization(EvaluateTree ev, ExperimentParameters exp_params, NeuralTree nTree) {
		this.mEvaluateTree = ev;
		this.mParams = exp_params;
		this.mTree = nTree;
	}

	public Individual optimize(File directory, String trial, String algoName) {
		System.out.println("Evaluating tree using Gradient Descent....");
		GD gd = new GD(mEvaluateTree, mParams, mTree, algoName);
		return gd.start(directory, trial);
	}
}
