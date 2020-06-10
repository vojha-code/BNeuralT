package neuralTree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import experimentSettings.ExperimentParameters;
import randoms.JavaRand;


/**
 * Function node in addition to its own members, inherits
 * m_edge_weight,m_parent_node, and some functions inherited from Node
 */
public class FunctionNode extends Node {
	int m_depth = 0; // initialise to be set by constructor -  current (self) depth
	int m_children = 0; // initialise to set by constructor - current (self) children
	ArrayList<Node> m_ChildrenList; // contain list of children nod of this (self list of children) node

	double m_bias = Double.POSITIVE_INFINITY; //bias node initialise to random value
	double m_delta_bias = Double.POSITIVE_INFINITY; // gradient of the node
	double m_delta_j = Double.POSITIVE_INFINITY; // function node delta_j for delta weight computation
	double m_activation = Double.POSITIVE_INFINITY; // output of this node

	ArrayList<Object> m_FunctionParams; // this fucntion's parameters

	/**
	 * Generating the Function node through its constructor
	 *
	 * @param p_weight      weigth of the edge connecting leaf node
	 * @param p_bias        :: this nodes bias weight
	 * @param p_fun_params  p_weight
	 * @param p_children    : number of children of this function node
	 * @param p_parent_node : its own parent node / for root its null
	 * @param p_depth       current depth of the function node
	 */
	public FunctionNode(double p_weight, double p_bias, ArrayList<Object> p_fun_params, int p_children, FunctionNode p_parent_node, int p_depth) {
		this.m_depth = p_depth;

		this.m_edge_weight = p_weight;
		this.m_bias = p_bias;
		this.m_children = p_children;
		this.m_ChildrenList = new ArrayList<Node>(); // clear list before appending to it.

		// at least three parameters a,b,c -> a and b parameters of the activation fun at the fun node, c is function type
		ArrayList<Object> n_fun_params = new ArrayList<Object>();
		n_fun_params.add((double) p_fun_params.get(0));
		n_fun_params.add((double) p_fun_params.get(1));
		n_fun_params.add(p_fun_params.get(2));
		this.m_FunctionParams = n_fun_params; // will create new objects

		this.m_parent_node = p_parent_node;
	}

	/**
	 * Generates a random tree. It is recursive procedure for tree generation.
	 * @param params
	 * :parm p_max_input_attr maximum number of input attributes
	 * :pram p_max_children maximum number of children one node can take
	 * :pram p_max_depth maximum depth of the whole tree
	 * :pram p_weight_range edge weight's range
	 * :pram m_current_depth current depth of the tree
	 * @param n_current_depth
	 */
	public void genrateChildren(ExperimentParameters params, int p_current_depth) {
		int p_max_target_attr = params.n_max_target_attr;
		int p_max_input_attr = params.n_max_input_attr;
		int p_max_children = params.n_max_children;
		int p_max_depth = params.n_max_depth;
		double[] p_weight_range = params.n_weight_range;
		double[] p_fun_range = params.n_fun_range;
		String p_fun_type = params.n_fun_type;
		String p_out_fun_type = params.n_out_fun_type;
		double p_probIntLeafNodeGen = params.n_probIntLeafNodeGen;

		JavaRand random = new JavaRand();

		if (p_current_depth < p_max_depth) {
			//iterate through number of children of this (current) function node
			for (int i = 0; i < m_children; i++) {
				double n_weight = random.nextUniform(p_weight_range[0], p_weight_range[1]); // tree edge weight between uniformly taken between 0 and 1
				double n_bias = random.nextUniform(p_weight_range[0], p_weight_range[1]); // tree bias weight between uniformly taken between 0 and 1
				int n_min_child = 2; // in any case min number of children will be 2 for a node

				ArrayList<Object> n_fun_params = new ArrayList<Object>();
				// a,b,c -> a and b parameters of the activation fun at the fun node, c is function type
				// only used for tanh and sigmoid
				n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1]));// unused for softmax
				n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1])); // unused for soft-max /tanh/sigmoid
				n_fun_params.add(p_fun_type); // this can be any thing Gaussing and tanh, sigm etc.

				// Child node generation function as well as leaf nodes
				if (p_max_target_attr > 1 && p_current_depth == 0) {
					// For multiple class problem - i.e., target has more than has one column (outputs)
					// We make sure all the child of roots are a function node and not a leaf node
					// generating random number of child for this function node
					int n_children_range = p_max_children - n_min_child; // compute range for randomise number of child for a node
					int n_children = n_min_child + random.nextInt(n_children_range); // random children number (ensure at-least 2 child)
					if (p_out_fun_type.equalsIgnoreCase("softmax")) {
						n_fun_params = new ArrayList<Object>();
						n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1]));// unused for softmax
						n_fun_params.add(random.nextUniform(0.1, 1.0)); // unused for softmax
						n_fun_params.add(p_out_fun_type); // this can be any thing Gaussing and tanh, sigm etc.
					}
					// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
					FunctionNode n_funNode = new FunctionNode(n_weight, n_bias, n_fun_params, n_children, this, p_current_depth); //parent of this node is self
					// recursion
					n_funNode.genrateChildren(params, p_current_depth + 1);
					m_ChildrenList.add(n_funNode);
				} else {
					// For single class (typically regression problem root can function as well as leaf nodes
					// a random choice between next node is leaf or a function
					// decision to generate a leaf node or a function node
					if (random.nextDouble() <= p_probIntLeafNodeGen) {//%
						// generate a leaf child of self as parent
						int n_num = random.nextInt(p_max_input_attr); // random number to determine a leaf node or a function  node child
						LeafNode n_leafnode = new LeafNode(n_weight, n_num, this);
						m_ChildrenList.add(n_leafnode);
					} else {//generate a function node
						// generate a function node (child) of the current node (self as a parent)
						// compute number of children for the new function node
						int n_children_range = p_max_children - n_min_child; // compute range for randomise number of child for a node
						int n_children = n_min_child + random.nextInt(n_children_range); // random children number (ensure at-least 2 child)
						// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
						FunctionNode n_funNode = new FunctionNode(n_weight, n_bias, n_fun_params, n_children, this, p_current_depth); //parent of this node is self
						// recursion
						n_funNode.genrateChildren(params, p_current_depth + 1);
						m_ChildrenList.add(n_funNode);
					} //End of if-else for probability decision
				} // End for multi-class problem
			} //end of for loop for all p_max_children
		} else {
			// generate only leaf node when max depth is reached
			// iterate through number of children of this function node
			for (int i = 0; i < m_children; i++) {
				double n_weight = random.random(p_weight_range[0], p_weight_range[1]); // tree edge weight between uniformly taken between 0 and 1
				// this generate random input attr number (p_max_input_attr is exclusive)
				int n_input_num = random.nextInt(p_max_input_attr);
				LeafNode n_leafnode = new LeafNode(n_weight, n_input_num, this); //  leaf node does not contain bias
				m_ChildrenList.add(n_leafnode);
			} //end for loop
		} //end if-else for depth of the three
	}// function

	/**
	 *
	 * @param p_tree NeuralTree
	 * @param depth height of the tree
	 */
	@Override
	public void inspect_node(NeuralTree p_tree, int p_depth) {
		try {
			for (Node node : m_ChildrenList) {
				// if print_message = True it print node type (except for root node) so toal print will one less that toatal nodes
				//boolean print_message = false;
				//Node node = (Node) m_Children1;
				if (node.isLeaf(false)) {
					node.inspect_node(p_tree, p_depth + 1);
					p_tree.addLeafNodesList((LeafNode) node);
				} else {
					node.inspect_node(p_tree, p_depth + 1);
					p_tree.addFunNodesList((FunctionNode) node);
				}
			}
		} catch (Exception e) {
			System.out.print("Error Inspection FuncNode" + e);
		}
	}//inspect node

	@Override
	public void display_node(int p_depth) {
		for (int i = 0; i < p_depth; i++) {
			System.out.print(" |");
		}
		System.out.println(" +" + m_children);

		for (Node node : m_ChildrenList) {
			//Node node = (Node) m_Children1;
			//System.out.print(" "+node.m_Weight);
			node.display_node(p_depth + 1);
		}
	}//end displayTree

	/**
	 * Printing function node child - or collecting them in a JSON format.
	 * :param p_depth    current depth of the tree
	 */
	@Override
	public String print_node(int p_depth) {
		String jason_string = "\"name\":" + "\"f " + (m_children) + "\",";

		jason_string = jason_string + "\"children\":[";

		int listLength = m_ChildrenList.size();
		int conut_loop = 0;
		for (Node node : m_ChildrenList) {
			//Node node = (Node) m_Children1;
			jason_string = jason_string + "{" + node.print_node(p_depth + 1);

			conut_loop = conut_loop + 1;
			if (conut_loop < listLength) {
				jason_string = jason_string + ","; // no comma for the last child in the list.
			}
		}
		jason_string = jason_string + " ]";

		return jason_string + "}";
	} // end printing function nodes

	/**
	 * Saving function node child - or collecting them in a JSON format.
	 * @param p_depth (int): current depth of the tree
	 * @return
	 */
	@Override
	public String saveNode(int p_depth) {
		String jason_string = "\"name\":" + "\"f:" + (m_children) + "; e:" + (m_edge_weight) + "; b:" + (m_bias) + "; p:" + (m_FunctionParams) + "\",";
		jason_string = jason_string + "\"children\":[";

		int listLength = m_ChildrenList.size();
		int conut_loop = 0;
		for (Node node : m_ChildrenList) {
			//Node node = (Node) m_Children1;
			jason_string = jason_string + "{" + node.saveNode(p_depth + 1);

			conut_loop = conut_loop + 1;
			if (conut_loop < listLength) {
				jason_string = jason_string + ","; // no comma for the last child in the list.
			}
		}
		jason_string = jason_string + " ]";
		return jason_string + "}";
	}

	// end printing function nodes

	/**
	 *
	 * @param p_nodes (FunctionNode)
	 * @return return a copy nodes
	 */
	@Override
	public Node copyNode(FunctionNode p_nodes) {
		// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
		FunctionNode node = new FunctionNode(m_edge_weight, m_bias, m_FunctionParams, m_children, p_nodes, m_depth);
		for (Node m_Children1 : m_ChildrenList) {
			//node.m_ChildrenList.add(((Node) m_Children1).copyNode(node));
			node.m_ChildrenList.add(m_Children1.copyNode(node));
		}
		return node;
	}//end copy node

	/**
	 * Retrieve children from the json file
	 * @param p_json_child_array
	 * @param p_depth
	 */
	public void retriveChildren(JSONArray p_json_child_array, int p_depth) {
		Iterator iterator = p_json_child_array.iterator();
		while (iterator.hasNext()) {
			JSONObject item = (JSONObject) iterator.next();
			//System.out.println("\n->"+item);
			//System.out.println("\n");
			if (item.containsKey("children")) {
				//generate function node
				JSONArray n_fun_children = (JSONArray) item.get("children"); // fetch the list of children
				int n_children = n_fun_children.size(); // this is the length of children list
				// n_children length can  be also found alternatively  using -> n_list_names position 0

				//Retrieving parameters
				String[] n_list_names = ((String) item.get("name")).split(";");
				//System.out.println(n_list_names.length+"\n"+n_list_names[0]);
				// n_list_names - > one is the weight
				double n_edge_weight = Double.parseDouble((n_list_names[1].split(":"))[1]);
				// names - > two is the bias
				double n_bias_weight = Double.parseDouble((n_list_names[2].split(":"))[1]);
				// names - > three is the function parameter
				String[] n_prams_val = (n_list_names[3].split(":"))[1].replaceAll("[\\[\\](){}]", "").split(",");
				ArrayList<Object> n_fun_params = new ArrayList<Object>();
				n_fun_params.add(Double.parseDouble(n_prams_val[0]));
				n_fun_params.add(Double.parseDouble(n_prams_val[1]));
				n_fun_params.add(n_prams_val[2].replace(" ", ""));

				// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
				FunctionNode n_funNode = new FunctionNode(n_edge_weight, n_bias_weight, n_fun_params, n_children, this, p_depth);
				n_funNode.retriveChildren(n_fun_children, p_depth + 1);
				// after all recursion the child list will be saved here
				m_ChildrenList.add(n_funNode);
			} else {
				// generate a leaf child of self as parent
				int index = 0; // 0 is index of attribute
				int edge = 1; // 1 is index of edge
				String[] leaf_values = ((String) item.get("name")).split(";");
				int n_num = (int) Double.parseDouble(leaf_values[index].split(":")[1]);
				double n_weight = Double.parseDouble(leaf_values[edge].split(":")[1]);
				LeafNode n_leafnode = new LeafNode(n_weight, n_num, this);
				m_ChildrenList.add(n_leafnode);
			}
		} //end while
	}//end retrieve tree

	/**
	 * Retrieving the current nodes all input attributes value
	 * @param out_noeds_inputs_attr
	 * @param node_count
	 * @return
	 */
	public ArrayList<Object> getLeafNodesValue(ArrayList<Integer> out_noeds_inputs_attr, List<Integer> node_count) {
		for (Node node : m_ChildrenList) {
			node_count.add(1);
			//if (((Node) node).isLeaf(false)) {
			if (node.isLeaf(false)) {
				out_noeds_inputs_attr.add(((LeafNode) node).getInputAttribute());
			} else {
				((FunctionNode) node).getLeafNodesValue(out_noeds_inputs_attr, node_count);
			}
		}
		ArrayList<Object> returnList = new ArrayList<Object>();
		returnList.add(out_noeds_inputs_attr);
		returnList.add(node_count);
		return returnList;
	}

	//Return edge weight of the node
	public double getBias() {
		return this.m_bias;
	}

	//Setting wedge weight
	public void setBias(double pBias) {
		m_bias = pBias;
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

	//Return edge weight of the node
	public double getDeltaBias() {
		return m_delta_bias;
	}

	//Return edge weight of the node
	@Override
	public double getDeltaEdgeWeight() {
		return m_delta_weight;
	}

	@Override
	public FunctionNode getParentNode() {
		return m_parent_node;
	}

	@Override
	public void setParentNode(FunctionNode parentNode) {
		m_parent_node = parentNode;
	}

	//---------------------------------------------------------------------------------------------//
	// TREE OUTPUT
	//---------------------------------------------------------------------------------------------//

	/**
	 * Function return tree output
	 * either the child nodes of the tree -  multi input single output (MISO) problem
	 * or the root node of the tree -  mulit input multi output (MIMO) problem
	 * @param p_input_attr_val  target columns number - equivalent to number of classes
	 * @return tree output  ArrayList
	 */
	public double[] getMultiNodeOutput(double[] p_input_vector, int p_target_col) {
		//MULITI OUTPUT - REGRSSION / CLASSIFICATION Problem for each child node
		//Evaluate the function nodes output for a given inputs data
		// iterate for all child nodes of this current node
		// for the first call it starts with all child nodes of the root.
		// so for classification problem all child node of the root will be taken
		double[] net_root_child_outputs = new double[p_target_col]; // list for the
		int i = 0;
		for (Node node : m_ChildrenList) {
			net_root_child_outputs[i] = ((FunctionNode) node).getSingleNodeOutput(p_input_vector);
			i++;
		}
		//return output for each class (each child of the root)
		return net_root_child_outputs;
	}

	/**
	 * Evaluate the function nodes output for a given inputs data
	 * @param p_input_attr_val input example vector
	 * @return
	 */
	@Override
	public double getSingleNodeOutput(double[] p_input_vector) {
		double net_node_wx = 0.0; // weighted summation
		// iterate for all child nodes of this current node
		for (Node node : m_ChildrenList) {
			//Node node = (Node) child;
			//weighted sum of all incoming values of the children
			// Its a recursion here -
			// IF the node is function it will call itself "getSingleNodeOutput()" in function_node.py
			// IF a leaf it will call leaf node "getSingleNodeOutput()" in leaf_node.py
			net_node_wx = net_node_wx + node.getEdgeWeight() * node.getSingleNodeOutput(p_input_vector);
		}
		// compute activation of the current node node based on its activation function
		//System.out.println("error"+net_node_wx);
		//directly using sigmoid to spped-up computation a bit
		double x = net_node_wx + m_bias; //wx + b
		if (x < 0.0) {
			m_activation = 1.0 - 1.0 / (1.0 + Math.exp(x));
		} else {
			m_activation = 1.0 / (1.0 + Math.exp(-x));
		}
		//ActivationFunction node_activation = new ActivationFunction(m_FunctionParams, net_node_wx, m_bias);
		//m_activation = node_activation.value();
		//m_activation = node_activation.sigmoid();//directly using sigmoid to speed-up computation a bit
		//System.out.println("Activation:"+ m_activation);
		return m_activation;
	}//END of single node output

	//---------------------------------------------------------------------------------------------//
	// GRADEIENT DESCENT
	//---------------------------------------------------------------------------------------------//
	/**
	 * Compute and sets the nodes delta j value that will we used to computer weight change
	 * @param y desired output
	 * @param ypred predicted output
	 * @param j
	 * @param isOutputNode set to false by default for hidden node the hidden function nodes takes false
	 */
	public void setNodeDeltaJ(double[] y, double[] ypred, int j, boolean isOutputNode) {
		if (isOutputNode) {
			m_delta_j = (ypred[j] - y[j]) * ypred[j] * (1.0 - ypred[j]);// Use this statement just for fast computation
			/*
			//OPEN THESE FOR Other function if in use
			//print('Out ',j,':', y[j], ypred[j], self.m_FunctionParams.__contains__('sigmoid'), end = ' ')
			if (m_FunctionParams.contains("softmax")) { // [ypred[j] -y[j]]*ypred[j]*ypred[j]
				m_delta_j = (ypred[j] - y[j]);
				//System.out.println("softmax gradient"); // did not perform as expected
			} else if (m_FunctionParams.contains("sigmoid")) { // (ypred[j] -y[j]]) * ypred[j]*(1.0 - ypred[j])
				m_delta_j = (ypred[j] - y[j]) * ypred[j] * (1.0 - ypred[j]);
			} else if (m_FunctionParams.contains("tanh")) { // -[ypred[j] -y[j]]*ypred[j]*ypred[j]
				m_delta_j = (-1.0 * (ypred[j] - y[j])) * ypred[j] * ypred[j];
			} */
		} else {
			// for all hidden nodes
			m_delta_j = m_activation * (1.0 - m_activation) * m_parent_node.m_delta_j * m_edge_weight;// Use this statement just for fast computation
			/*
			//OPEN THESE FOR Other function
			double hj = m_activation; //  the nodes own activation
			double delta_k = m_parent_node.m_delta_j; //  delata of the higher/ paren nodes
			double wjk = m_edge_weight; //  weight leads to the parent node (uper/higher layer)
			//print('H ', hj, delta_k, wjk, self.m_FunctionParams.__contains__('sigmoid'), end = ' ')
			if (m_FunctionParams.contains("sigmoid")) { //  yj[1.0-yj]*deltak*wk
				m_delta_j = hj * (1.0 - hj) * delta_k * wjk;
			} else if (m_FunctionParams.contains("tanh")) { //  -yj*yj*deltak*wk
				m_delta_j = (-1.0 * hj * hj) * delta_k * wjk;
			} */
		}
		//print('del_j', self.m_delta_j)
		for (Node node : m_ChildrenList) {
			//Node node = (Node) child;
			if (!node.isLeaf(false)) {
				((FunctionNode) node).setNodeDeltaJ(null, null, -1, false);// null because non output node
			}
		}
	}//set node delta ends

	/**
	 * for function node gradient of weight has its own activation as the input
	 * @param p_input_attr_val
	 * @param both_w_n_b
	 */
	@Override
	public void setNodeGradient(double[] p_input_attr_val, boolean both_w_n_b) {
		if (both_w_n_b) {
			// FOR NON OUTPUT Function nodes (for leaf node check in leaf function)

			//double yi = m_activation; // activation of the current node act as an input for the next nodes (parent node)
			//double del_j = m_parent_node.m_delta_j; //  delta_j of parent node backproagate to previous layer
			//print('gradient: ',yi, del_j)
			//m_delta_weight = del_j * yi;
			m_delta_weight = m_parent_node.m_delta_j * m_activation;//values used directly to reduce memory and speedup computation
			m_delta_bias = m_delta_j;
		} else {
			// FOR OUTPUT function nodes
			m_delta_bias = m_delta_j;
		}
	}//set Node gradient ends

	//---------------------------------------------------------------------------------------------//
	// GENTIC OPERATORS
	//---------------------------------------------------------------------------------------------//

	/**
	 * replace subtree of the parent node
	 * @param oldNode FunctionNode
	 * @param newNode FunctionNode
	 */
	public void replaceSubTree(FunctionNode oldNode, FunctionNode newNode) {// delete oldNode?
		try {
			//System.out.print(m_Children.size()+": "+m_Children.contains(oldNode)+": "+m_Children.indexOf(oldNode));
			if (m_ChildrenList.contains(oldNode)) {
				m_ChildrenList.set(m_ChildrenList.indexOf(oldNode), newNode); //remove old and replace by newNode
				//System.out.println("subtree is replaced");
			} else {
				//System.out.print("no replacements");
			}
		} catch (Exception e) {
			System.out.print("Error Replaccing:" + e);
		}
	}//end relapse

	/**
	 * Replace a function node by a leaf node
	 * @param p_fun_node_to_replace
	 * @param by_leaf_node
	 */
	public void removeAndReplace(FunctionNode p_fun_node_to_replace, LeafNode by_leaf_node) {
		try {
			//System.out.print(m_Children.size()+": "+m_Children.contains(node)+": "+m_Children.indexOf(node));
			if (m_ChildrenList.contains(p_fun_node_to_replace)) {
				m_ChildrenList.set(m_ChildrenList.indexOf(p_fun_node_to_replace), by_leaf_node);
				//System.out.println(" "+m_Children.size()+": "+m_Children.contains(node));
			} else {
				//System.out.print("no replacements");
			}
		} catch (Exception e) {
			System.out.print("Error RemoveAndReplace-" + e);
		}
	}//end replace

	/**
	 * Replace leaf node by and growing a subtree at its place
	 * @param p_leaf_node_to_replace
	 * @param params set of parameters
	 */
	public void removeAndGrow(LeafNode p_leaf_node_to_replace, ExperimentParameters params) {
		try {
			//System.out.print(m_Children.size()+": "+m_Children.contains(toRemove)+": "+m_Children.indexOf(toRemove));
			if (m_ChildrenList.contains(p_leaf_node_to_replace)) {
				int p_max_input_attr = params.n_max_input_attr;
				int p_max_children = params.n_max_children;
				double[] p_weight_range = params.n_weight_range;
				double[] p_fun_range = params.n_fun_range;
				String p_fun_type = params.n_fun_type;
				JavaRand random = new JavaRand();

				double n_weight = random.nextUniform(p_weight_range[0], p_weight_range[1]); // tree edge weight between uniformly taken between 0 and 1
				double n_bias = random.nextUniform(p_weight_range[0], p_weight_range[1]); // tree bias weight between uniformly taken between 0 and 1
				int n_min_child = 2; // in any case min number of children will be 2 for a node

				ArrayList<Object> n_fun_params = new ArrayList<Object>();
				n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1]));// unused for softmax
				n_fun_params.add(random.nextUniform(0.1, 1.0)); // unused for softmax
				n_fun_params.add(p_fun_type); // this can be any thing Gaussing and tanh, sigm etc.

				// generating random number of child for this function node
				int n_children_range = p_max_children - n_min_child; // compute range for randomise number of child for a node
				int n_children = n_min_child + random.nextInt(n_children_range); // random children number (ensure at-least 2

				// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
				FunctionNode n_fun_node = new FunctionNode(n_weight, n_bias, n_fun_params, n_children, this, m_depth); //parent of this node is self
				// recursion
				n_fun_node.genrateChildren(params, m_depth + 1);
				//replace the leaf by a sub-tree
				m_ChildrenList.set(m_ChildrenList.indexOf(p_leaf_node_to_replace), n_fun_node);
			} else {
				System.out.print("no remove and Grow");
			}
		} catch (Exception e) {
			System.out.print("Error RemoveAndGraow-" + e);
		}
	}// end Replace leaf node
}//End Function node
