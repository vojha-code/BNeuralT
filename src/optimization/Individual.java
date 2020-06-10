package optimization;

import java.util.ArrayList;

import neuralTree.NeuralTree;

//An individual in NSGA population
public class Individual {

	public NeuralTree mTree = null;

	public double[] mCost = null;
	public double[] mNormalizedCost = null;

	public int mRank;
	public ArrayList<Integer> mDominationSet = null;
	public int mDominatedCount;

	public int mAssociatedRef;
	public double mDistanceToAssociatedRef;

	public ArrayList<Object> mCostAll;
}
