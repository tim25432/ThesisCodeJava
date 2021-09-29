import java.util.Arrays;


/**
 * Class used to model a layer of a DNN
 * @author Tim Tjhay (495230tt)
 */
public class Layer implements Cloneable {
	private int k;
	private int n;
	private double[][] weights;
	private double[] bias;
	
	private double[] upperBoundsX;
	private double[] lowerBoundsX;
	
	private double[] upperBoundsS;
	private double[] lowerBoundsS;
	
	/**
	 * Initializes the layer with the weights and biases
	 * @param k			The index of the layer(which layer it is)
	 * @param n_k		The number of neurons in the layer
	 * @param weights	The weights
	 * @param bias		The biases
	 */
	public Layer(int k, int n_k, double[][] weights, double[] bias) {
		// store the relevant data
		this.k = k;
		this.n = n_k;
		this.weights = weights;
		this.bias = bias;
		
		// set the bounds to (-inf, inf)
		this.lowerBoundsX = new double[n_k];
		this.lowerBoundsS = new double[n_k];
		
		this.upperBoundsX = new double[n_k];
		Arrays.fill(this.upperBoundsX, Integer.MAX_VALUE);
		this.upperBoundsS = new double[n_k];
		Arrays.fill(this.upperBoundsS, Integer.MAX_VALUE);
	}
	
	/**
	 * Method used to create a deep copy of a layer 
	 */
	public Layer clone() throws CloneNotSupportedException{
		// create a copy of the layer
		Layer clone = (Layer)super.clone();
		
		// copy the arrays individually so original will not 
		// be affected by changes to clone
		if (this.weights != null) {
			clone.weights = new double[this.weights.length][];
			int n_k_1 = this.weights[0].length;
			for (int i=0; i < this.weights.length; i++) {
				clone.weights[i] = Arrays.copyOf(this.weights[i], n_k_1);
			}

			clone.bias = Arrays.copyOf(this.bias, this.n);
		}

		clone.lowerBoundsS = Arrays.copyOf(this.lowerBoundsS, this.n);
		clone.lowerBoundsX = Arrays.copyOf(this.lowerBoundsX, this.n);
		clone.upperBoundsS = Arrays.copyOf(this.upperBoundsS, this.n);
		clone.upperBoundsX = Arrays.copyOf(this.upperBoundsX, this.n);
		
		return clone;
	}
	
	/**
	 * Method that returns index of the layer
	 * @return	The index of the layer
	 */
	public int getK() {
		return k;
	}
	
	/**
	 * Method that returns the number of neurons in the layer
	 * @return	The number of neurons in the layer
	 */
	public int getN() {
		return this.n;
	}
	
	/**
	 * Method that returns the weights corresponding to this layer
	 * @return	The weights corresponding to this layer
	 */
	public double[][] getWeights() {
		return this.weights;
	}
	
	/**
	 * Method that returns the biases corresponding to this layer
	 * @return	The biases corresponding to this layer
	 */
	public double[] getBias() {
		return this.bias;
	}
	
	/**
	 * Method that returns the upper bounds of the x variables for the neurons in the layer
	 * @return Upper bounds on x variables of neurons
	 */
	public double[] getUpperBoundsX() {
		return this.upperBoundsX;
	}

	/**
	 * Method that sets the upper bounds on the x variables of this layer
	 * @param upperBounds	New upper bounds on x variables
	 */
	public void setUpperBoundsX(double[] upperBounds) {
		this.upperBoundsX = upperBounds;
	}
	
	/**
	 * Method that returns the lower bounds of the x variables for the neurons in the layer
	 * @return Lower bounds on x variables of neurons
	 */
	public double[] getLowerBoundsX() {
		return this.lowerBoundsX;
	}

	/**
	 * Method that sets the lower bounds on the x variables of this layer
	 * @param lowerBounds	New lower bounds on x variables
	 */
	public void setLowerBoundsX(double[] lowerBounds) {
		this.lowerBoundsX = lowerBounds;
	}

	/**
	 * Method that returns the upper bounds of the s variables for the neurons in the layer
	 * @return upper bounds on s variables of neurons
	 */
	public double[] getUpperBoundsS() {
		return this.upperBoundsS;
	}

	/**
	 * Method that sets the upper bounds on the x variables of this layer
	 * @param upperBounds	New upper bounds on x variables
	 */
	public void setUpperBoundsS(double[] upperBounds) {
		this.upperBoundsS = upperBounds;
	}
	
	/**
	 * Method that returns the lower bounds of the s variables for the neurons in the layer
	 * @return Lower bounds on s variables of neurons
	 */
	public double[] getLowerBoundsS() {
		return this.lowerBoundsS;
	}

	/**
	 * Method that sets the lower bounds on the s variables of this layer
	 * @param lowerBounds	New lower bounds on s variables
	 */
	public void setLowerBoundsS(double[] lowerBounds) {
		this.lowerBoundsS = lowerBounds;
	}
	
}
