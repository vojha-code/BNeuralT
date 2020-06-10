package neuralTree;

import java.util.ArrayList;

public class Node {
	double m_edge_weight = Double.POSITIVE_INFINITY; //edge weight of the node
	double m_delta_weight = Double.POSITIVE_INFINITY; // gradient of the node
	FunctionNode m_parent_node; // the parent node of the node

	public Node() {
	}

	public Node(double p_weight, FunctionNode p_parent_node) {
		this.m_edge_weight = p_weight;
		this.m_parent_node = p_parent_node;
		//System.out.println("parent ->"+parentNode);
	}

	void inspect_node(NeuralTree p_tree, int p_depth) {
		System.out.println("Overide inspect_node methods");
	}

	boolean isLeaf(boolean p_toPrintNodeType) {
		if (p_toPrintNodeType) {
			System.out.println("Function node");
		}
		return false; // return false
	}

	String print_node(int p_depth) {
		System.out.println("Overide this method: print_node");
		return null;// this is ok!
	}

	void display_node(int depth) {
		System.out.println("Should be override");
	}

	String saveNode(int p_depth) {
		System.out.println("Overide this method: saveNode");
		return null;// this is ok!
	}

	Node copyNode(FunctionNode parentNode) {
		System.out.println("This should be override Node");
		return null;// this is ok!
	}

	double getSingleNodeOutput(double[] p_input_attr_val) {
		//Output of a single node
		System.out.println("Override default - regression output method");
		return -999.000; //Intentionally given a rubbish value because it will override
	}

	ArrayList<Double> getMultiNodeOutput(double[] p_input_attr_val) {
		//Output of multiple nodes
		System.out.println("Override: default - class output method");
		return null; // Intentionally given a rubbish value because it will override
	}

	//Setting weight change
	void setNodeGradient(double[] p_input_attr_val, boolean both_w_n_b) {
		this.m_delta_weight = -999.000;
	}

	//*******************************************************************//
	//          Common methods of function node and leaf node            //
	//*******************************************************************//

	//Return edge weight of the node
	public double getEdgeWeight() {
		return this.m_edge_weight;
	}

	//Setting wedge weight
	public void setEdgeWeight(double pEdgeWeight) {
		this.m_edge_weight = pEdgeWeight;
	}

	//Return edge weight of the node
	public double getDeltaEdgeWeight() {
		return this.m_delta_weight;
	}

	FunctionNode getParentNode() {
		return this.m_parent_node;
	}

	public void setParentNode(FunctionNode parentNode) {
		m_parent_node = parentNode;
	}

}// End Node
