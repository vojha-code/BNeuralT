package neuralTree;

import java.util.ArrayList;

import randoms.JavaRand;

//This calss implements activation function for the nodes
public class ActivationFunction {
	ArrayList<Object> m_function_param; // List of parameters of a function
	double m_net_wighted_sum = -99.99; // Set an arbitrary weighed sum at the node
	double m_bias = -99.99; // Set an arbitrary weighed sum at the node
	JavaRand random = new JavaRand();

	/**
	 *  Contractor that sets the param values
	 * @param p_function_param ArrayList<Object>  is the parameter of the defined function
	 * @param p_net_weighted_sum is the value on which activating/quashing/transfer function is to be applied.
	 * @param p_bias bia of a node
	 */
	public ActivationFunction(ArrayList<Object> p_function_param, double p_net_weighted_sum, double p_bias) {
		this.m_function_param = p_function_param;
		this.m_net_wighted_sum = p_net_weighted_sum;
		this.m_bias = p_bias;
	}

	//Implementation of of different functions
	public double value() {
		//return sigmoid();
		if (m_function_param.contains("Gaussian")) {
			//System.out.println("Using Gaussian");
			return Gaussian();
		}
		if (m_function_param.contains("tanh")) {
			//System.out.println("Using tanh");
			return tanh();
		}
		if (m_function_param.contains("sigmoid")) {
			//System.out.println("Using sigmoid");
			return sigmoid();
		}
		if (m_function_param.contains("ReLU")) {
			//System.out.println("Using ReLU");
			return ReLU();
		}
		if (m_function_param.contains("softmax")) {
			//System.out.println("Softmax function should be processed outside Actiovation class");
			return m_net_wighted_sum + m_bias; // did not perfrom since each out node is an independent model
		} else {
			System.out.println("No function found");
			return -99.99;// Set an arbitrary weighed sum at the node
		}

		/*String activationFun = (String) m_function_param.get(2);
		switch (activationFun) {
		//check what function is at the node
		case "sigmoid":
			//System.out.println("Using sigmoid");
			return sigmoid();

		case "Gaussian":
			//System.out.println("Using Gaussian");
			return Gaussian();

		case "tanh":
			//System.out.println("Using tanh");
			return tanh();

		case "ReLU":
			//System.out.println("Using ReLU");
			return ReLU();

		case "softmax":
			System.out.println("Softmax function should be outside Actiovation class");
			System.exit(0);
			return -99.99;// Set an arbitrary weighed sum at the node

		default:
			System.out.println("No function found");
			System.exit(0);
			return -99.99;// Set an arbitrary weighed sum at the node
		} */
	}

	private double ReLU() {
		//Implementation of ReLU function
		// ReLU function is from:
		//    https://en.wikipedia.org/wiki/Sigmoid_function
		//Define the content
		//double wx = m_net_wighted_sum;
		//double b = m_bias;
		double x = m_net_wighted_sum + m_bias;  //wx + b
		return Math.max(0, x);
	}

	public double sigmoid() {
		//  Implementation of sigmoid function
		//  sigmoid function is from:
		//      https://en.wikipedia.org/wiki/Sigmoid_function

		//Define the content
		//double wx = m_net_wighted_sum;
		//double b = m_bias;
		double x = m_net_wighted_sum + m_bias; //wx + b
		if (x < 0.0) {
			return 1 - 1 / (1 + Math.exp(x));
		}
		return 1.0 / (1.0 + Math.exp(-x));
	}

	private double tanh() {
		// Implementation of tanh function
		// tanh function is from:
		//     http://mathworld.wolfram.com/HyperbolicTangent.html
		//Define the constent
		//double wx = m_net_wighted_sum;
		//double b = m_bias;
		double x = m_net_wighted_sum + m_bias; //wx + b
		//System.out.print(" wx:"+wx+" b:"+b+" x:"+x+" ");
		return Math.tanh(x);
		//return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x));
	}

	private double Gaussian() {
		//Implementation of Gausian function
		//Guassian function is from:
		//    https://docs.scipy.org/doc/scipy/reference/stats.html
		// Node that Guassian function has no use of bias / can use a bais

		//Define the content
		double x = m_net_wighted_sum;
		double mu = (double) m_function_param.get(0);
		double sigma = (double) m_function_param.get(1);
		if (sigma == 0.0) {
			sigma = random.nextUniform(0.1, 1.0); // this was the error
			m_function_param.set(1, sigma);
		}
		// See definition of Gaussian here: https://en.wikipedia.org/wiki/Gaussian_function
		x = (x - mu) / sigma; // e -(1/2)((x-mu)/sigma)^2
		return Math.exp(-(x * x) / 2.0) / Math.sqrt(2.0 * Math.PI) / sigma;
	}

	public static void main(String[] args) {
		//Tested Ok!
		double wx = 2.2;
		double b = 0.05;
		double mu = 0.25;
		double sigma = 0.8;
		String[] f = { "Gaussian", "sigmoid", "tanh", "ReLU" };

		for (String fval : f) {
			ArrayList<Object> p_function_param = new ArrayList<Object>();
			p_function_param.add(mu);
			p_function_param.add(sigma);
			p_function_param.add(fval);

			ActivationFunction act = new ActivationFunction(p_function_param, wx, b);
			System.out.println(fval + " val : " + act.value());
		}
		/*
			0.02556584221597675
			0.02556584221597675

			0.9046505351008906
			0.9046505351008906

			0.9780261147388136
			0.9780261147388136*/

	}
}
//End of class
