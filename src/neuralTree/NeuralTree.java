package neuralTree;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;

import org.ejml.data.DenseMatrix64F;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.JSONParser;

import experimentSettings.ExperimentParameters;
import randoms.JavaRand;


public class NeuralTree {
	//class tree takes
	public FunctionNode m_root; // initialise root as a function node
	int m_depth; //depth/height of the tree (including root level) initialise to zero
	double m_treeFitness; // Arbitrary value to set fitness
	int m_max_children;

	ArrayList<FunctionNode> FunNodesList; // list of function nodes
	ArrayList<LeafNode> LeafNodesList; // list of leaf nodes

	public NeuralTree() {
		this.m_root = null; // initialise root node
		this.FunNodesList = new ArrayList<FunctionNode>(); // list of function nodes
		this.LeafNodesList = new ArrayList<LeafNode>(); // list of leaf nodes
		this.m_max_children = 0;
		this.m_depth = 0;
		this.m_treeFitness = 999.0; // arbitrary  value to set fitness
	}// initiate Neural Tree

	/**
	 *
	 * @param params HashMap<Object>
	 * 	Generates a random tree
	    :param    m_max_input_attr     maximum number of input variable (attributes) in the data set
	    :param    m_max_children       maximum number of child one node can takes
	    :param    m_max_depth          maximum depth/height of the tree
	 */
	public void genrateGenericRandomTree(ExperimentParameters params) {
		//int p_max_target_attr = (int) params.get("n_max_target_attr");
		int p_max_target_attr = params.n_max_target_attr;
		int p_max_children = params.n_max_children;
		double[] p_weight_range = params.n_weight_range;
		double[] p_fun_range = params.n_fun_range;
		String p_fun_type = params.n_fun_type;
		String p_out_fun_type = params.n_out_fun_type;

		double n_root_weight = -1.0; // Fix weight for root node (not to  be used anyway - it's only for Function Node to work)
		double n_root_bias = 1.0; // Fix bias for root node (not to  be used anyway - it's only for Function Node to work)
		int n_min_child = 2; // setting minimum nodes for a node in the tree is 2
		int n_arity_range = p_max_children - n_min_child; // compute range for randomise number of child for a node

		JavaRand random = new JavaRand();
		// Check number of outputs of the problem
		int n_children = 0;
		if (p_max_target_attr > 1) {
			n_children = p_max_target_attr;// fix the number of child of a tree for a classification problem
		} else {
			n_root_bias = random.nextUniform(p_weight_range[0], p_weight_range[1]); // for regression problems bias weight is required
			// random children number (ensure at-least 2 child)
			n_children = n_min_child + random.nextInt(n_arity_range);
		}

		// a,b,c -> a and b parameters of the activation fun at the fun node, c is function type
		// Function params are mostly used for Guassian function
		ArrayList<Object> n_fun_params = new ArrayList<Object>();
		if (p_out_fun_type.equalsIgnoreCase("softmax")) {
			n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1]));// unused for softmax
			n_fun_params.add(random.nextUniform(0.1, 1.0)); // unused for softmax /tanh/sigmoid
			n_fun_params.add(p_out_fun_type); // this can be any thing Gaussing and tanh, sigm etc.
		} else {
			n_fun_params.add(random.nextUniform(p_fun_range[0], p_fun_range[1]));// unused for softmax
			n_fun_params.add(random.nextUniform(0.1, 1.0)); // unused for softmax /tanh/sigmoid
			n_fun_params.add(p_fun_type); // this can be any thing Gaussing and tanh, sigm etc. but a softmax
		}

		// Intialize root node of the tree  and set parent as none
		int n_current_depth = -1; // Set depth as -1 for root node since root do not have a parent node - None
		// Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
		m_root = new FunctionNode(n_root_weight, n_root_bias, n_fun_params, n_children, m_root, n_current_depth);

		n_current_depth = 0; // re-set current depth (root node depth)  to zero
		// Generate children of the tree - generate the whole tree in recursive manner
		m_root.genrateChildren(params, n_current_depth);
		inspectChildNodes(); // calling a function within the class to inspect
	}//End definition of generic Random tree

	/**
	 * Collecting objects of the tree
	 */
	public void inspectChildNodes() {
		// TODO Auto-generated method stub
		FunNodesList.clear();
		LeafNodesList.clear();
		m_depth = 0;
		m_root.inspect_node(this, 0);// set the root root depth to zero.
	}

	/**
	 * Adding leaf node in a vector of leaf nodes
	 * @param leafNode leaf node of the tree
	 */
	public void addLeafNodesList(LeafNode leafNode) {
		//System.out.println("leaf child");
		LeafNodesList.add(leafNode);
	}

	/**
	 * Returning the list of leaf
	 * @return leaf child nodes list
	 */
	public ArrayList<LeafNode> getLeafNodesList() {
		return LeafNodesList;
	}

	/**
	 * Adding function node in a vector of function nodes
	 * @param functNode function node of the tree
	 */
	public void addFunNodesList(FunctionNode functNode) {
		//System.out.println("fun child");
		FunNodesList.add(functNode);
	}

	/**
	 * Returning the list of function nodes
	 * @return return function node child list
	 */
	public ArrayList<FunctionNode> getFunNodesList() {
		return FunNodesList;
	}

	public ArrayList<Node> getChildrenList() {
		return m_root.m_ChildrenList;
	}

	/**
	 * Set height of the tree
	 * @param depth height of the tree
	 */
	public void setDepth(int p_depth) {
		m_depth = p_depth;
	}

	/**
	 * Returns height of the tree
	 * @return height of the tree
	 */
	public int getDepth() {
		return m_depth;
	}

	/**
	 * set the fitness of the tree
	 * (depend on the objective function user RMSE of Error Rate)
	 * @param p_fitness (double) RMSE or Error rate of tree for a training data
	 */
	public void setTreeFitness(double p_fitness) {
		m_treeFitness = p_fitness;
	}

	/**
	 * return fitness of the tree
	 * (depend on the objective function user RMSE of Error Rate)
	 * @return double:  RMSE of Error Rate of this tree
	 */
	public double getTreeFitness() {
		return m_treeFitness;
	}

	/**
	 * return total number of nodes in the tree (i.e a root node have)
	 * @return double: number nodes
	 */
	public int getTreeSize() {
		return (FunNodesList.size() + LeafNodesList.size() + 1); //+ 1 for root node
	}

	/**
	 * return total number of nodes in the tree (i.e a root node have)
	 * @return double: number nodes
	 */
	public int getFuncNodeSize() {
		return FunNodesList.size(); //+ 1 for root node
	}

	/**
	 * return total number of nodes in the tree (i.e a root node have)
	 * @return double: number nodes
	 */
	public int getLeafNodeSize() {
		return LeafNodesList.size(); //+ 1 for root node
	}

	/**
	 * Display tree outline
	 */
	public void displayTree() {
		m_root.display_node(0);
	}// End definition display_node

	/**
	 * Printing only the outline of the tree
	 * @return (String) print the tree in JSON format - It will create a dictionary
	 */
	public String saveTreeOutline() {
		return "{ " + m_root.print_node(0); // returns a JSON string
	}// End definition print_tree

	/**
	 * Save full the outline of the tree
	 * @return (String) Saving the tree in JSON format - It will create a dictionary
	 */
	public String saveTreeModel() {
		return "{ " + m_root.saveNode(0); // returns a JSON string
	}// End definition print_tree

	/**
	 * Retrieve the tree in JSON format - It will create a dictionary
	 * @param p_json_tree
	 * @return NeuralTree
	 */
	public void retriveJsonTree(JSONObject p_json_tree) {
		JSONArray n_root_children = (JSONArray) p_json_tree.get("children"); // _NUL if key not present
		String[] n_list_names = ((String) p_json_tree.get("name")).split(";");
		//System.out.println(n_list_names.length+"\n"+ Arrays.toString(n_list_names));
		if (n_list_names.length < 4 || n_list_names.length > 4) {
			System.out.println("Please input a correct tree mode file treeModel.json");
		} else {
			// names - > zero is number of children
			//int n_children = int(n_list_names[0].split(':')[1])
			// Alternatively can number of children taken from children size
			int n_children = n_root_children.size();
			//System.out.println("Number of child: "+n_children);
			// names - > one is the weight
			double n_root_weight = Double.parseDouble((n_list_names[1].split(":"))[1]);
			// names - > two is the bias
			double n_root_bias = Double.parseDouble((n_list_names[2].split(":"))[1]);
			// names - > three is the function parameter
			String[] n_prams_val = (n_list_names[3].split(":"))[1].replaceAll("[\\[\\](){}]", "").split(",");
			ArrayList<Object> n_fun_params = new ArrayList<Object>();
			n_fun_params.add(Double.parseDouble(n_prams_val[0]));
			n_fun_params.add(Double.parseDouble(n_prams_val[1]));
			n_fun_params.add(n_prams_val[2].replace(" ", ""));
			//System.out.println("\ne:"+n_root_weight+" b:"+n_root_bias+" p:"+n_fun_params.toString());

			int n_current_depth = -1; // Set depth as -1 for root node
			// set parent None: // Function Node = WEIGHT, BIAS, FUN_PARMS, //CHILDREN, PARENT, DEPTH
			m_root = new FunctionNode(n_root_weight, n_root_bias, n_fun_params, n_children, m_root, n_current_depth);

			n_current_depth = 0; //
			m_root.retriveChildren(n_root_children, n_current_depth);
			inspectChildNodes(); // calling a function within the class to inspect
			System.out.println("Tree Scussessfully read");
		}
	}// End retrieve tree

	/**
	 * return a copy of itself with new instance / reference to objects
	 * @return
	 */
	public NeuralTree copy_Tree() {
		NeuralTree n_tree = new NeuralTree();
		n_tree.m_max_children = m_max_children;
		n_tree.m_depth = m_depth;
		n_tree.m_treeFitness = m_treeFitness;
		n_tree.m_root = (FunctionNode) m_root.copyNode(m_root);
		n_tree.inspectChildNodes();
		return n_tree;
	}

	/**
	 *
	 * @param p_target_attr_count
	 * @param input_featuer_names
	 * @param outputs_names
	 * @return
	 */
	public HashMap<String, Object> getTreeInputFeatuereProperties(int p_target_attr_count, String[] input_featuer_names, String[] outputs_names) {
		ArrayList<Integer> root_node_input_attr_all = new ArrayList<Integer>();
		for (LeafNode node : getLeafNodesList()) {
			root_node_input_attr_all.add(node.getInputAttribute());
		}
		List<Integer> root_node_unique_featuers = new ArrayList<Integer>(new HashSet<>(root_node_input_attr_all));
		Collections.sort(root_node_unique_featuers);

		ArrayList<String> root_node_unique_featuers_names = new ArrayList<String>();
		ArrayList<Integer> root_node_each_featuers_count = new ArrayList<Integer>();
		for (Integer input_idx : root_node_unique_featuers) {
			root_node_unique_featuers_names.add(input_featuer_names[input_idx.intValue()]);
			root_node_each_featuers_count.add(Collections.frequency(root_node_input_attr_all, input_idx));
		}
		//System.out.println(root_node_input_attr_all);
		//System.out.println(root_node_unique_featuers);
		//System.out.println(root_node_unique_featuers_names);
		//System.out.println(root_node_each_featuers_count);

		ArrayList<Object> each_out_nodes_feature_all = new ArrayList<Object>();
		ArrayList<Object> each_out_nodes_feature_unique = new ArrayList<Object>();
		ArrayList<Object> each_out_nodes_feature_unique_names = new ArrayList<Object>();
		ArrayList<Object> each_out_nodes_each_feature_count = new ArrayList<Object>();
		ArrayList<Integer> each_out_nodes_tree_size = new ArrayList<Integer>();

		if (p_target_attr_count == 1) {
			each_out_nodes_feature_all.add(root_node_input_attr_all);
			each_out_nodes_feature_unique.add(root_node_unique_featuers);
			each_out_nodes_feature_unique_names.add(root_node_unique_featuers_names);
			each_out_nodes_each_feature_count.add(root_node_each_featuers_count);
			each_out_nodes_tree_size.add(getTreeSize());
		} else {
			for (Node node : m_root.m_ChildrenList) {
				ArrayList<Integer> out_noeds_inputs_attr = new ArrayList<Integer>();
				List<Integer> node_count = new ArrayList<Integer>();
				//out_noeds_inputs_attr node_count =
				ArrayList<Object> rternedList = ((FunctionNode) node).getLeafNodesValue(out_noeds_inputs_attr, node_count);
				out_noeds_inputs_attr = (ArrayList<Integer>) rternedList.get(0);
				node_count = (ArrayList<Integer>) rternedList.get(0);

				// collecting nodes in a list
				each_out_nodes_feature_all.add(out_noeds_inputs_attr);
				List<Integer> this_node_unique_featuers = new ArrayList<Integer>(new HashSet<>(out_noeds_inputs_attr));
				Collections.sort(out_noeds_inputs_attr);
				each_out_nodes_feature_unique.add(this_node_unique_featuers);
				ArrayList<String> this_node_unique_featuers_names = new ArrayList<String>();
				ArrayList<Integer> this_node_each_featuers_count = new ArrayList<Integer>();
				for (Integer input_idx : this_node_unique_featuers) {
					this_node_unique_featuers_names.add(input_featuer_names[input_idx.intValue()]);
					this_node_each_featuers_count.add(Collections.frequency(out_noeds_inputs_attr, input_idx));
				}
				each_out_nodes_feature_unique_names.add(this_node_unique_featuers_names);
				each_out_nodes_each_feature_count.add(this_node_each_featuers_count);
				each_out_nodes_tree_size.add(node_count.size() + 2);
			}
		} //end checking all out nodes
		HashMap<String, Object> tree_feature_properties = new HashMap<String, Object>();
		tree_feature_properties.put("tree_out_nodes", Arrays.asList(outputs_names));
		tree_feature_properties.put("tree_featueres_all", root_node_input_attr_all);
		tree_feature_properties.put("tree_featueres_unique", root_node_unique_featuers);
		tree_feature_properties.put("tree_featueres_unique_names", root_node_unique_featuers_names);
		tree_feature_properties.put("tree_each_featuere_count", root_node_each_featuers_count);
		tree_feature_properties.put("tree_size_overall", getTreeSize());
		tree_feature_properties.put("each_out_node_featueres_all", each_out_nodes_feature_all);
		tree_feature_properties.put("each_out_node_featueres_unique", each_out_nodes_feature_unique);
		tree_feature_properties.put("each_out_node_featueres_unique_names", each_out_nodes_feature_unique_names);
		tree_feature_properties.put("each_out_node_each_featueres_count", each_out_nodes_each_feature_count);
		tree_feature_properties.put("each_out_node_size", each_out_nodes_tree_size);
		return tree_feature_properties;
	}//tree properties retrieved

	/**
	 * Retrieving tree parameters
	 * @param p_target_attr_count how many target attributes
	 * number of output class. for classification it should be 2 or more.
	 * For regression MUST be 1
	 * @param pRerival retiring all values weights only
	 * @return Returns a vector of edge and function parameters values
	 */
	public double[] getTreeParameters(int p_target_attr_count, String pRerival) {
		double[] n_Parameters = null;
		int nParameterLength = -1;
		int indx;
		if (pRerival.equalsIgnoreCase("all")) {//  retrieve Weight, Bias, and Function params
			if (p_target_attr_count == 1) {
				if (m_root.m_FunctionParams.contains("Gaussian")) {
					nParameterLength = (2 + FunNodesList.size() * 3) + LeafNodesList.size();//root 2parms +  func's (weight and 2params) + leaf weight
				} else {
					nParameterLength = (1 + FunNodesList.size() * 2) + LeafNodesList.size();//root bias +  func's (weight and bias) + leaf weight
				}
			} else {
				if (m_root.m_FunctionParams.contains("Gaussian")) {
					nParameterLength = (FunNodesList.size() * 3 - p_target_attr_count) + LeafNodesList.size();// func's (weight and 2params) - no immediate root's child weights + leaf weight
				} else {
					nParameterLength = (FunNodesList.size() * 2 - p_target_attr_count) + LeafNodesList.size();// func's (weight and bias) - no immediate root's child weights + leaf weight
				}
			}
			indx = 0;
			n_Parameters = new double[nParameterLength];

			// For single output problem - ROOT NODE's parameters (Bias and Function_params) are useful (not weight)
			//  SINGLE OUTPUT Else all root node's parameters are useless
			if (p_target_attr_count == 1) {
				// NO WEIGHT because root is output node and its weight has no use
				ArrayList<Object> funParams = m_root.m_FunctionParams; // TAKE ROOT FUN PARAMS
				if (!funParams.contains("Gaussian")) {
					//System.out.print("param get" + funParams.get(3) + " " + nParameterLength);
					n_Parameters[indx] = m_root.getBias(); // TAKE ROOT BIAS
					indx++;
				} else {
					//System.out.print("param get" + nParameterLength);
					for (int i = 0; i < funParams.size() - 1; i++) {
						n_Parameters[indx] = (double) funParams.get(i); // adding a and b of fun params
						indx++;
					}
				}
			} //for single output end

			//  FUNCTION NODES - Retrieve weight, bias, and func parameters [a and b] from the function list
			for (FunctionNode node : FunNodesList) {
				if (p_target_attr_count == 1) {
					// For single output problem, root node is the output node
					// Hence all its children function nodes are hidden nodes. This their weights are useful
					n_Parameters[indx] = node.getEdgeWeight();// take all hidden function node's (roots any child node) weights
					indx++;
				} else {
					// For multi output problem root node is useless and all roots immediate child nodes are function nodes
					// Hence take ONLY NON-ROOT child weights
					if (!m_root.m_ChildrenList.contains(node)) {// NOT a ROOT's immediate child
						n_Parameters[indx] = node.getEdgeWeight();// TAKE WEIGHTS ONLY for NON OUTPUT NODE
						indx++;
					}
				}

				// check if node is NOT Gausian if NOT -> take ONLY bias Else take ONLY function param
				ArrayList<Object> funParams = node.m_FunctionParams; // TAKE NODE's FUN PARAMS
				if (!funParams.contains("Gaussian")) {
					//System.out.print("param get" + funParams.get(2) + " " + nParameterLength);
					// Retrieving Function bias
					n_Parameters[indx] = node.getBias();// TAKE NODE's BIAS
					indx++;
				} else {
					//System.out.print(" param get" + funParams.get(2) + " " + nParameterLength);
					// Retrieving function params
					for (int i = 0; i < funParams.size() - 1; i++) {
						n_Parameters[indx] = (double) funParams.get(i);// adding a and b of fun params
						indx++;
					}
				} //end Gaussian function check
			} //end all function node check

			//  LEAF NODES - retrieve weight and parameters a, b from the function list
			for (LeafNode node : LeafNodesList) {
				n_Parameters[indx] = node.getEdgeWeight();
				indx++;
			}
			//end all param collection
		} else if (pRerival.equalsIgnoreCase("weights_and_bias")) {
			if (p_target_attr_count == 1) {
				nParameterLength = (1 + FunNodesList.size() * 2) + LeafNodesList.size();// root bias - yes/ root weights no
			} else {
				nParameterLength = (FunNodesList.size() * 2 - p_target_attr_count) + LeafNodesList.size();// no immediate root's child weights
			}

			indx = 0;
			n_Parameters = new double[nParameterLength];

			if (p_target_attr_count == 1) {
				// NO WEIGHT only // TAKE ROOT BIAS
				n_Parameters[indx] = m_root.getBias();
				indx++;
			}
			// Retrieve function node weights and parameters a, b from the function list
			for (FunctionNode node : FunNodesList) {
				if (p_target_attr_count == 1) {
					n_Parameters[indx] = node.getEdgeWeight();// take all hidden function node's (roots any child node) weights
					indx++;
				} else {
					if (!m_root.m_ChildrenList.contains(node)) {// NOT a ROOT's immediate child
						n_Parameters[indx] = node.getEdgeWeight();// TAKE WEIGHTS ONLY for NON OUTPUT NODE's child
						indx++;
					}
				}
				// Retrieving bias for all function nodes
				n_Parameters[indx] = node.getBias();// TAKE NODE's BIAS
				indx++;
			}
			// Retrieve leaf node weights - it has no bias
			for (LeafNode node : LeafNodesList) {
				n_Parameters[indx] = node.getEdgeWeight();
				indx++;
			}
		} else if (pRerival.equalsIgnoreCase("weights")) {
			if (p_target_attr_count == 1) {
				nParameterLength = (FunNodesList.size()) + LeafNodesList.size();// no root weight
			} else {
				nParameterLength = (FunNodesList.size() - p_target_attr_count) + LeafNodesList.size(); // no immediate root's child weights
			}
			indx = 0;
			n_Parameters = new double[nParameterLength];
			// Retrieve function node weights and parameters a, b from the function list
			for (FunctionNode node : FunNodesList) {
				// Retrieving weights
				if (p_target_attr_count == 1) {// for regression problem take all its child weights
					n_Parameters[indx] = node.getEdgeWeight();
					indx++;
				} else {// if multi output  problem only take non roots child weights
					if (!m_root.m_ChildrenList.contains(node)) {// NOT IN ROOT CHILD -> ok
						n_Parameters[indx] = node.getEdgeWeight();// TAKE WEIGHTS ONLY for NON OUTPUT NODE
						indx++;
					}
				}
			}
			// Retrieve leaf node weights
			for (LeafNode node : LeafNodesList) {
				n_Parameters[indx] = node.getEdgeWeight();
				indx++;
			}
		} else if (pRerival.equalsIgnoreCase("bias")) {
			if (p_target_attr_count == 1) {
				nParameterLength = 1 + FunNodesList.size(); //root bias - yes
			} else {
				nParameterLength = FunNodesList.size(); // all nodes bias yes
			}
			indx = 0;
			n_Parameters = new double[nParameterLength];
			if (p_target_attr_count == 1) {
				// NO WEIGHT only // TAKE ROOT BIAS
				n_Parameters[indx] = m_root.getBias();
				indx++;
			}
			// Retrieve function node weights and parameters a, b from the function list
			for (FunctionNode node : FunNodesList) {
				// Retrieving bias for all function nodes
				n_Parameters[indx] = node.getBias();// TAKE NODE's BIAS
				indx++;
			}
		} else {
			System.out.println("please input one of these: all, weights_and_bias,  weights,  bias");
		}
		//return accumulated parameters;
		//double[] dDList = nParameters.stream().mapToDouble(Double::doubleValue).toArray();
		//return dDList;
		return n_Parameters;
	}//getParamter Done

	/**
	 * Set tree edges with the vector pEdgeWeightsVec
	 * @param pParameters double[]
	 * @param p_target_attr_count int
	 * @param pSet String
	 */
	public void setTreeParameters(double[] pParameters, int p_target_attr_count, String pSet) {
		if (pSet.equalsIgnoreCase("all")) { // set Weight, Bias, and Function params
			int indx = 0;
			// setting roots function parameters only for regression problem
			if (p_target_attr_count == 1) { //only useful for regression
				ArrayList<Object> funParams = m_root.m_FunctionParams; // Check ROOT FUN PARAMS
				if (!funParams.contains("Gaussian")) {
					// ROOT  NODE's bias
					m_root.setBias(pParameters[indx]); // SET ROOT's BIAS
					indx += 1;
				} else {
					System.out.print("param set");
					//parameter only useful in Gausssan function
					for (int i = 0; i < funParams.size() - 1; i++) {
						funParams.set(i, pParameters[indx]); // adding a and b of fun params
						indx += 1;
					}
				}
			}
			// set function node weights and parameters a, b from the function list
			for (FunctionNode node : FunNodesList) {
				// SET nodes weights
				if (p_target_attr_count == 1) {
					// For single output problem child weights were taken so will be set
					node.setEdgeWeight(pParameters[indx]);
					indx += 1;
				} else {
					// For multi output problem problem only non roots child weights were taken
					// so only non roots child weight will be set
					if (!m_root.m_ChildrenList.contains(node)) {// SET Weight only for node NOT IN ROOT CHILD -> ok
						node.setEdgeWeight(pParameters[indx]);
						indx += 1;
					}
				}

				// check if node is NOT Gausian
				ArrayList<Object> funParams = node.m_FunctionParams;
				if (!funParams.contains("Gaussian")) {
					// SET nodes bias
					node.setBias(pParameters[indx]); // SET ROOT BIAS
					indx += 1;
				} else {
					System.out.print("param set");
					//SET nodes function params
					for (int i = 0; i < funParams.size() - 1; i++) {
						funParams.set(i, pParameters[indx]); // adding a and b of fun params
						indx += 1;
					}
				}
			}
			// set leaf node weights
			for (LeafNode node : LeafNodesList) {
				node.setEdgeWeight(pParameters[indx]);
				indx += 1;
			}
		} // all ends
		if (pSet.equalsIgnoreCase("weights_and_bias")) { // set Weight, Bias, and Function params
			int indx = 0;
			if (p_target_attr_count == 1) {
				// ROOT  NODE's bias and parameter only useful in regression problems
				m_root.setBias(pParameters[indx]); // SET ROOT's BIAS
				indx += 1;
			}
			// set node weights
			for (FunctionNode node : FunNodesList) {
				// SET nodes weights
				if (p_target_attr_count == 1) {
					// For single output problem child weights were taken so will be set
					node.setEdgeWeight(pParameters[indx]);
					indx += 1;
				} else {
					// For multi output problem problem only non roots child weights were taken
					// so only non roots child weight will be set
					if (!m_root.m_ChildrenList.contains(node)) {// SET Weight only for node NOT IN ROOT CHILD -> ok
						node.setEdgeWeight(pParameters[indx]);
						indx += 1;
					}
				}
				// SET nodes bias
				node.setBias(pParameters[indx]); // SET nodes BIAS
				indx += 1;
			}
			// set leaf node weights
			for (LeafNode node : LeafNodesList) {
				node.setEdgeWeight(pParameters[indx]);
				indx += 1;
			}
		} //end weights_and_bias

		if (pSet.equalsIgnoreCase("weights")) { // set Weight, Bias, and Function params
			int indx = 0;
			// set node weights
			for (FunctionNode node : FunNodesList) {
				// SET nodes weights
				if (p_target_attr_count == 1) {
					// For single output problem child weights were taken so will be set
					node.setEdgeWeight(pParameters[indx]);
					indx += 1;
				} else {
					// For multi output problem problem only non roots child weights were taken
					// so only non roots child weight will be set
					if (!m_root.m_ChildrenList.contains(node)) {// SET Weight only for node NOT IN ROOT CHILD -> ok
						node.setEdgeWeight(pParameters[indx]);
						indx += 1;
					}
				}
			}
			// set leaf node weights
			for (LeafNode node : LeafNodesList) {
				node.setEdgeWeight(pParameters[indx]);
				indx += 1;
			}
		} //end weights
		if (pSet.equalsIgnoreCase("bias")) { // set Weight, Bias, and Function params
			int indx = 0;
			if (p_target_attr_count == 1) {
				// ROOT  NODE's bias and parameter only useful in regression problems
				m_root.setBias(pParameters[indx]); // SET ROOT's BIAS
				indx += 1;
			}
			// set node weights
			for (FunctionNode node : FunNodesList) {
				// SET nodes bias
				node.setBias(pParameters[indx]); // SET nodes BIAS
				indx += 1;
			}
		} //end weights_and_bias
	}//end setting values

	/**
	 * Evaluate tree output for a given inputs data
	 * @param p_input_attr_val input example vector
	 * @param p_target_attr_count target columns number - equivalent to number of classes
	 * @return ArrayList<Double>
	 */
	public double[] getOutput(double[] p_input_vector, int p_target_attr_count) {
		//System.out.println("\n Get output.....");
		double[] outputs = new double[p_target_attr_count];
		if (p_target_attr_count > 1) {
			//MULITI OUTPUT - REGRSSION / CLASSIFICATION Problem for each child node
			//Evaluate the function nodes output for a given inputs data
			if (p_target_attr_count == m_root.m_ChildrenList.size()) {
				outputs = m_root.getMultiNodeOutput(p_input_vector, p_target_attr_count);
				/*
				//OPEN These for more general tree
				if (m_root.m_FunctionParams.contains("softmax")) {
					outputs = softmax(m_root.getMultiNodeOutput(p_input_vector, p_target_attr_count));
				} else {
					outputs = m_root.getMultiNodeOutput(p_input_vector, p_target_attr_count);
					//System.out.println("outputs T:" + Arrays.toString(outputs));
					//outputs = multi_out.stream().mapToDouble(Double::doubleValue).toArray();
				} */
			} else {
				System.out.println("Class number and root's child list doesnot match");
			}
		} else {
			//SINGLE OUTPUT For regression / classification and normal node output use this function
			//Evaluate the function nodes output for a given inputs data
			//System.out.println("Single attr");
			outputs[0] = m_root.getSingleNodeOutput(p_input_vector);
		}
		return outputs;
	}// end get Output

	private double[] softmax(double[] x) {
		//double[] x = multiNodeOutput.stream().mapToDouble(Double::doubleValue).toArray();
		int len = x.length;
		double[] exps = new double[len];
		double exps_sum = 0.0;
		//System.out.println("X value:       "+Arrays.toString(x));
		for (int i = 0; i < len; i++) {
			exps[i] = Math.exp(x[i]);
			exps_sum += exps_sum + exps[i];
		}
		for (int i = 0; i < len; i++) {
			exps[i] = exps[i] / exps_sum;

		}
		//System.out.println("Softmax value: "+Arrays.toString(exps));
		return exps;
	}

	public double[] getGradient(double[] x, double[] y, int p_target_attr_count) {
		double[] ypred = getOutput(x, p_target_attr_count);
		//print('y_pred: ', ypred, y, type(ypred), type(y))
		//if (p_target_attr_count > 1):
		//    ypred = softmax(ypred)
		setDeltaJ(y, ypred, p_target_attr_count);
		setGradient(x, p_target_attr_count);
		return getDeltaWeightandBias(p_target_attr_count);
	}//end get Gradient

	public DenseMatrix64F getGradientDense(double[] x, double[] y, int p_target_attr_count) {
		double[] ypred = getOutput(x, p_target_attr_count);
		//print('y_pred: ', ypred, y, type(ypred), type(y))
		//if (p_target_attr_count > 1):
		//    ypred = softmax(ypred)
		setDeltaJ(y, ypred, p_target_attr_count);
		setGradient(x, p_target_attr_count);
		return getDeltaWeightBiasDense(p_target_attr_count);
	}//end get Gradient

	/**
	 *
	 * @param p_target_attr_count  total class
	 * @return double[] gradient of the tree weights
	 */
	public double[] getDeltaWeightandBias(int p_target_attr_count) {
		ArrayList<Double> nGrad = new ArrayList<Double>();
		if (p_target_attr_count == 1) {
			// NO WEIGHT only // TAKE ROOT BIAS
			nGrad.add(m_root.getDeltaBias());
		}
		// Retrieve function node weights and parameters a, b from the function list
		for (FunctionNode node : FunNodesList) {
			if (p_target_attr_count == 1) {
				nGrad.add(node.getDeltaEdgeWeight());// // take all hidden function node's (roots any child node) weights
			} else {
				if (!m_root.m_ChildrenList.contains(node)) {// NOT a ROOT's immediate child
					nGrad.add(node.getDeltaEdgeWeight()); // TAKE WEIGHTS ONLY for NON OUTPUT NODE's child
				}
			}
			// Retrieving bias for all function nodes
			nGrad.add(node.getDeltaBias()); // TAKE NODE's BIAS
		} //for all  fun nodes
		// Retrieve leaf node weights - it has no bias
		for (LeafNode node : LeafNodesList) {
			nGrad.add(node.getDeltaEdgeWeight());
		}
		return nGrad.stream().mapToDouble(Double::doubleValue).toArray();
	}//end get delta weight

	/**
	 * get Delta Weight and Bias in Dense Matrix form
	 * @param p_target_attr_count
	 * @return  DenseMatrix64F a vector of size [ 1 x #Param]
	 */
	public DenseMatrix64F getDeltaWeightBiasDense(int p_target_attr_count) {
		DenseMatrix64F n_Parameters = null;
		int nParameterLength = -1;
		int indx;
		if (p_target_attr_count == 1) {
			nParameterLength = (1 + FunNodesList.size() * 2) + LeafNodesList.size();// root bias - yes/ root weights no
		} else {
			nParameterLength = (FunNodesList.size() * 2 - p_target_attr_count) + LeafNodesList.size();// no immediate root's child weights
		}

		indx = 0;
		n_Parameters = new DenseMatrix64F(1, nParameterLength);

		if (p_target_attr_count == 1) {
			// NO WEIGHT only // TAKE ROOT BIAS
			n_Parameters.set(indx, m_root.getDeltaBias());
			indx++;
		}
		// Retrieve function node weights and parameters a, b from the function list
		for (FunctionNode node : FunNodesList) {
			if (p_target_attr_count == 1) {
				n_Parameters.set(indx, node.getDeltaEdgeWeight());// take all hidden function node's (roots any child node) weights
				indx++;
			} else {
				if (!m_root.m_ChildrenList.contains(node)) {// NOT a ROOT's immediate child
					n_Parameters.set(indx, node.getDeltaEdgeWeight());// TAKE WEIGHTS ONLY for NON OUTPUT NODE's child
					indx++;
				}
			}
			// Retrieving bias for all function nodes
			n_Parameters.set(indx, node.getDeltaBias());// TAKE NODE's BIAS
			indx++;
		}
		// Retrieve leaf node weights - it has no bias
		for (LeafNode node : LeafNodesList) {
			n_Parameters.set(indx, node.getDeltaEdgeWeight());
			indx++;
		}
		return n_Parameters;
	}//end getWeight and bias

	/**
	 * get Weight and Bias in Dense Matrix form
	 * @param p_target_attr_count
	 * @return  DenseMatrix64F a vector of size [ 1 x #Param]
	 */
	public DenseMatrix64F getWeightBiasDense(int p_target_attr_count) {
		DenseMatrix64F n_Parameters = null;
		int nParameterLength = -1;
		int indx;
		if (p_target_attr_count == 1) {
			nParameterLength = (1 + FunNodesList.size() * 2) + LeafNodesList.size();// root bias - yes/ root weights no
		} else {
			nParameterLength = (FunNodesList.size() * 2 - p_target_attr_count) + LeafNodesList.size();// no immediate root's child weights
		}

		indx = 0;
		n_Parameters = new DenseMatrix64F(1, nParameterLength);

		if (p_target_attr_count == 1) {
			// NO WEIGHT only // TAKE ROOT BIAS
			n_Parameters.set(indx, m_root.getBias());
			indx++;
		}
		// Retrieve function node weights and parameters a, b from the function list
		for (FunctionNode node : FunNodesList) {
			if (p_target_attr_count == 1) {
				n_Parameters.set(indx, node.getEdgeWeight());// take all hidden function node's (roots any child node) weights
				indx++;
			} else {
				if (!m_root.m_ChildrenList.contains(node)) {// NOT a ROOT's immediate child
					n_Parameters.set(indx, node.getEdgeWeight());// TAKE WEIGHTS ONLY for NON OUTPUT NODE's child
					indx++;
				}
			}
			// Retrieving bias for all function nodes
			n_Parameters.set(indx, node.getBias());// TAKE NODE's BIAS
			indx++;
		}
		// Retrieve leaf node weights - it has no bias
		for (LeafNode node : LeafNodesList) {
			n_Parameters.set(indx, node.getEdgeWeight());
			indx++;
		}
		return n_Parameters;
	}//end getWeight and bias

	/**
	 * Set directly from the dense vector (just to save some computation time)
	 * @param pParameters
	 * @param p_target_attr_count
	 */
	public void setWeightBaisDense(DenseMatrix64F pParameters, int p_target_attr_count) {
		int indx = 0;
		if (p_target_attr_count == 1) {
			// ROOT  NODE's bias and parameter only useful in regression problems
			m_root.setBias(pParameters.get(indx)); // SET ROOT's BIAS
			indx += 1;
		}
		// set node weights
		for (FunctionNode node : FunNodesList) {
			// SET nodes weights
			if (p_target_attr_count == 1) {
				// For single output problem child weights were taken so will be set
				node.setEdgeWeight(pParameters.get(indx));
				indx += 1;
			} else {
				// For multi output problem problem only non roots child weights were taken
				// so only non roots child weight will be set
				if (!m_root.m_ChildrenList.contains(node)) {// SET Weight only for node NOT IN ROOT CHILD -> ok
					node.setEdgeWeight(pParameters.get(indx));
					indx += 1;
				}
			}
			// SET nodes bias
			node.setBias(pParameters.get(indx)); // SET nodes BIAS
			indx += 1;
		}
		// set leaf node weights
		for (LeafNode node : LeafNodesList) {
			node.setEdgeWeight(pParameters.get(indx));
			indx += 1;
		}
	}//end set parameters of dense

	/**
	 * Setting delta_j of each function node of the tree
	 * @param x inputs input vector
	 * @param p_target_attr_count  target columns number - equivalent to number of classes
	 */
	private void setGradient(double[] x, int p_target_attr_count) {
		if (p_target_attr_count == 1) {
			// No need to use gradient of weight for the output node but gradient of bias is necessary
			m_root.setNodeGradient(null, false); // for output nodes we set only bias -so set both w and b = False
			//print('delta_j',self.m_root.m_delta_j)
		}
		// For all function node of the root node
		for (FunctionNode node : FunNodesList) {
			//print('delta_j',node.m_parent_node.m_delta_j)
			if (p_target_attr_count == 1) {
				// for single output problem, all child (of root) takes weight and bias.
				// Hence we set delta_weight and delta_bias
				node.setNodeGradient(null, true);
			} else {
				// if multi output problem only set non roots child take weights and bias.
				// Hence we set delta weights and delta_bias
				if (!m_root.m_ChildrenList.contains(node)) {// NOT IN ROOT CHILD -> ok
					// for NON OUTPUT nodes delta_weight and delta_bias are necessary
					node.setNodeGradient(null, true);
				} else {
					// for OUTPUT nodes only delta_bias are necessary
					node.setNodeGradient(null, false); // for output nodes we set only bias -set set both w and b = False
				}
			}
		}
		// For all leaf node only weights are necessary
		for (LeafNode node : LeafNodesList) {
			node.setNodeGradient(x, false);
		}
	}

	/**
	 * Setting delta_j of each function node of the tree
	 * @param y  output vector (target)
	 * @param ypred tree output vector
	 * @param p_target_attr_count target columns number - equivalent to number of classes
	 */
	private void setDeltaJ(double[] y, double[] ypred, int p_target_attr_count) {
		if (p_target_attr_count == 1) {
			// Root node delta j is only useful for single output problems
			// print('root delta: ')
			m_root.setNodeDeltaJ(y, ypred, 0, true);
		} else {
			//System.out.println("-------------------evaluate gradient");
			for (Node child : m_root.m_ChildrenList) {
				FunctionNode node = (FunctionNode) child;
				//print('root ',  self.m_root.m_ChildrenList.index(node),' delta: ')
				node.setNodeDeltaJ(y, ypred, m_root.m_ChildrenList.indexOf(child), true);
			} //all roots child
		}
	}//end setDeltaj

	/**
	 *
	 * @param p_tree NeuralTree
	 * @param directory path to folder File type
	 * @param uniquefileName
	 * @param p_target_count
	 * @param input_names
	 * @param outputs_names
	 * @param isAll
	 */
	public void saveTreeModel(NeuralTree p_tree, File directory, String uniquefileName, int p_target_count, String[] input_names, String[] outputs_names, boolean isAll) {
		/* Open this if you want a separate folder for the trees
		directory = new File(directory + File.separator + "tree");
		if (!directory.exists()) {
			directory.mkdir();
		}*/

		//Saving a Json file of tree model
		try {
			FileWriter fwModel, fwOutline, fwPrpoertiese;
			fwModel = new FileWriter(directory + File.separator + uniquefileName + "_model.json");
			fwModel.write(p_tree.saveTreeModel());
			fwModel.close();

			if (isAll) {
				fwOutline = new FileWriter(directory + File.separator + uniquefileName + "_outline.json");
				fwOutline.write(p_tree.saveTreeOutline());
				fwOutline.close();

				HashMap<String, Object> tree_properties = p_tree.getTreeInputFeatuereProperties(p_target_count, input_names, outputs_names);
				JSONObject json = new JSONObject();
				json.putAll(tree_properties);
				//System.out.println(tree_properties);
				//System.out.println(json.toJSONString());
				fwPrpoertiese = new FileWriter(directory + File.separator + uniquefileName + "_properties.json");
				fwPrpoertiese.write(json.toJSONString());
				fwPrpoertiese.close();

				replaceTreeView(p_tree.saveTreeOutline(), directory, uniquefileName);
			}
		} catch (IOException e1) {
			System.out.print("i/o: tree save file closed");
			//e1.printStackTrace();
		}

	}//save tree model

	private void replaceTreeView(String p_tree_outline, File directory, String uniquefileName) {
		String oldTreeData = "var treeData";
		String newTreeData = "var treeData = [" + p_tree_outline + "];";

		String filename_org = ExperimentParameters.modelPath + File.separator + "view" + File.separator + "tree_view_org.html";
		FileWriter fwView;
		try {
			fwView = new FileWriter(directory + File.separator + uniquefileName + "_view.html");
			Scanner htmlLines = new Scanner(new File(filename_org));
			int line_number = 1;
			while (htmlLines.hasNextLine()) {
				String line = htmlLines.nextLine();
				if (line.contains(oldTreeData)) {
					//System.out.println(" Model Outline Saved - Check replaced HTML line: 48? = line replaced is " + line_number);
					fwView.write(newTreeData + "\n");
				} else {
					fwView.write(line + "\n");
				}
				line_number++;
			}
			fwView.close();
			htmlLines.close();
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}// end replaceTreeView

	/**
	 * Read a tree from file
	 * @param directory
	 * @param fileNme
	 * @return NeuralTree
	 */
	public void readTreeModel(File directory, String fileNme) {
		//NeuralTree nt = new NeuralTree();
		JSONParser parser = new JSONParser();
		try {
			Object jsonStringobj = JSONValue.parse(new FileReader(directory + File.separator + fileNme));
			JSONObject jsonObject = (JSONObject) jsonStringobj;
			retriveJsonTree(jsonObject);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//return nt;
	}// read file complete

}// End Neural Tree
