package neuralTree;

public class LeafNode extends Node {

	int m_input_attr = -99; // Initialise arbitrary to be set by function/constructor

	/**
	 * constructor for Leaf node
	 * @param p_weight weigth of the edge connecting leaf node
	 * @param p_input_attr the input attribute to be considered at the leaf node
	 * @param p_parent_node parent node of the leaf node
	 */
	public LeafNode(double p_weight, int p_input_attr, FunctionNode p_parent_node) {
		this.m_edge_weight = p_weight;
		this.m_input_attr = p_input_attr;
		this.m_parent_node = p_parent_node;
	}//end of definition generateChildren

	/**
	 * return boolean - whether it is a leaf node
	 */
	@Override
	public boolean isLeaf(boolean p_toPrintNodeType) {
		/*if (p_toPrintNodeType) {
			System.out.println("Leaf node");
		}*/
		return true;
	}// returns leaf verification

	/**
	/**
	 * Inspects current function node
	 * @param p_tree NeuralTree
	 * @param depth height of the tree
	 */
	@Override
	public void inspect_node(NeuralTree p_tree, int p_depth) {
		if (p_depth > p_tree.getDepth()) {
			p_tree.setDepth(p_depth);
		}
	}// end inspect node (leaf)

	@Override
	public void display_node(int depth) {
		for (int i = 0; i < depth; i++) {
			System.out.print(" -");
		}
		System.out.println(" :" + m_input_attr);
	}

	/**
	 * Printing only leaf nodes (returning leaf node string)
	 * :param p_depth    current depth of the tree
	 */
	@Override
	public String print_node(int p_depth) {
		return "\"name\":" + "\" i " + (m_input_attr) + "\" }";
	}

	/**
	 * Saving only leaf nodes (returning leaf node string)
	 */
	@Override
	public String saveNode(int p_depth) {
		return "\"name\":" + "\" i:" + (m_input_attr) + "; e:" + (m_edge_weight) + "\" }";
	}

	/**
	 *
	 * @param p_node (FunctNode)
	 * @return return leaf node copy with its parent node being p_node
	 */
	@Override
	public Node copyNode(FunctionNode p_node) {
		// new object of leaf node
		LeafNode node = new LeafNode(m_edge_weight, m_input_attr, p_node);
		return node;
	}

	//return index of the inputs attribute
	public int getInputAttribute() {
		return m_input_attr;
	}

	public void setInputAttribute(int p_input_rand_attr) {
		m_input_attr = p_input_rand_attr;
	}

	//Return edge weight of the node
	@Override
	public double getEdgeWeight() {
		return m_edge_weight;
	}

	//Setting wedge weight
	@Override
	public void setEdgeWeight(double pEdgeWeight) {
		m_edge_weight = pEdgeWeight;
	}

	@Override
	public FunctionNode getParentNode() {
		return m_parent_node;
	}

	@Override
	public void setParentNode(FunctionNode parentNode) {
		m_parent_node = parentNode;
	}

	/**
	 *Return value of the value at the index m_input_attr from the leaf node
	 * @param p_input_attr_val input vector
	 * @return
	 */
	@Override
	public double getSingleNodeOutput(double[] p_input_vector) {
		return p_input_vector[m_input_attr];
	}

	/**
	 * for Leaf node node gradient of weight has its own input value as the input
	 * @param p_input_attr_val
	 * @param both_w_n_b
	 */
	@Override
	public void setNodeGradient(double[] p_input_attr_val, boolean both_w_n_b) {
		m_delta_weight = m_parent_node.m_delta_j * p_input_attr_val[m_input_attr];// just to make it faseter
		/*double xi = p_input_attr_val[m_input_attr]; //  input to the next layer
		double del_j = m_parent_node.m_delta_j; //  delta_j of the next layer (gradient of the parent node)
		m_delta_weight = del_j * xi; */
	}

}//End Leaf class
