package optimization;

import java.util.ArrayList;

import neuralTree.EvaluateTree;
import neuralTree.NeuralTree;

public class CostFunction {

	EvaluateTree pEvaluation;
	int mObj;

	/**
	 *
	 * @param pEvaluation data to evaluate
	 * @param mObj  defult is 2 for nsga-2 and nsga-3
	 */
	public CostFunction(EvaluateTree p_Evaluation, int p_Obj) {
		this.pEvaluation = p_Evaluation;
		this.mObj = p_Obj;
	}

	/**
	 * Compute a vector of objective functions values
	 * @param pTree  the tree on which to evaluate
	 * @param pSet type of dataset
	 * @return
	 */
	public double[] costOnlyObjectives(ArrayList<Object> nCostAll) {
		double[] nCost = new double[mObj];
		nCost[0] = (double) nCostAll.get(0);
		nCost[1] = ((Number) nCostAll.get(nCostAll.size() - 1)).doubleValue();
		return nCost;

		/* Open these for single objective problems implementation
		if (mObj == 1) {
			double[] nCost = new double[mObj];
			nCost[0] = (double) nCostAll.get(0);
			return nCost;
		} else {
			double[] nCost = new double[mObj];
			nCost[0] = (double) nCostAll.get(0);
			for (int j = 1; j < mObj; j++) {
				nCost[j] = ((Number) nCostAll.get(nCostAll.size() - j)).doubleValue();
			}
			return nCost;
		}*/
	}//end cost

	/**
	 * Compute a vector of objective functions values
	 * @param pTree  the tree on which to evaluate
	 * @param pSet type of dataset
	 * @return All cost function values
	 */
	public ArrayList<Object> costAll(NeuralTree pTree, String pSet) {
		ArrayList<Object> cost = new ArrayList<Object>();
		pEvaluation.set_dataset_to_evaluate(pSet);

		// Sequential compute
		pEvaluation.getTreePredictedOutputs(pTree); // collected output prediction not used here

		// Parallel compute
		//pEvaluation.getTreePredictedOutputsParallel(pTree);

		// Parallel GPU compute - yet to be implemented
		//pEvaluation.getTreePredictedOutputsGPU(pTree);

		cost.addAll(pEvaluation.getTreeFitness());
		cost.add(pTree.getTreeSize());
		//more objectives for NSGA should be added here.
		//System.out.println(cost);
		return cost;
	}
}
