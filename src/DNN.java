import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

import ilog.concert.IloException;


/**
 * Class used to model a DNN
 * @author Tim Tjhay (495230tt)
 */
public class DNN {
	private Layer[] layers;
	private int K;
	
	/**
	 * Initializes the DNN by reading the weights from the file
	 * @param filename	Name of the file containing the weights
	 * @param K			The number of layers the DNN has(excluding the input layer)
	 * @throws FileNotFoundException
	 */
	public DNN(String filename, int K) throws FileNotFoundException {
		// create the layers
		this.layers = new Layer[K + 1];
		this.K = K;
		
		// read the weights
		readWeights(filename);
	}
	
	/**
	 * Initializes a DNN using an array of layers
	 * @param layers
	 */
	public DNN(Layer[] layers) {
		this.layers = layers;
		// exclude the input layer
		this.K = layers.length - 1;
	}
	
	/**
	 * Method that returns the number of layers the DNN has
	 * @return	Number of layers in the DNN
	 */
	public int getNLayers() {
		return this.K;
	}
	
	/**
	 * Returns an array of the layers in the DNN
	 * @return	An array of the layers in the DNN
	 */
	public Layer[] getLayers() {
		return this.layers;
	}
	
	/**
	 * Method used to calculate and set the bounds of the neurons
	 * @param timeLimit		If a time limit should be imposed on the bound tightening(weaker improved model)
	 * @throws CloneNotSupportedException
	 * @throws IloException
	 */
	public void calculateBounds(boolean timeLimit) throws CloneNotSupportedException, IloException {
		// for all layers except the input layer:
		for (int k=1; k < this.layers.length; k++) {
			// create an array to store the previous layers
			Layer[] layers = new Layer[k+1];
			
			// add copies of the previous layers
			for (int i=0; i < k; i++) {
				layers[i] = this.layers[i].clone();
			}
			
			Layer layer = this.layers[k];
			
			// create arrays to store the upper bounds
			double[] xUB = new double[layer.getN()];
			double[] sUB = new double[layer.getN()];
			
			// get the weights and biases of this layer
			double[][] weights = layer.getWeights();
			double[] bias = layer.getBias();
			// for every neuron in the layer:
			for (int j=0; j < layer.getN(); j++) {
				// get the weights and bias corresponding to this neuron
				double[][] w_j = {weights[j]};
				double[] b_j = {bias[j]};
				// create a layer containing only this neuron
				Layer layer_j_k = new Layer(k, 1, w_j, b_j);
				layers[k] = layer_j_k;
				
				// create a DNN using the copied previous layers and this new layer
				DNN dnnBounds = new DNN(layers);
				
				// get and store the bounds
				MILPBounds boundsModel = new MILPBounds(dnnBounds, timeLimit);
				double[] bounds = boundsModel.getUpperBounds();
				boundsModel.cleanup();
				
				xUB[j] = bounds[0];
				sUB[j] = bounds[1];
			}
			layer.setUpperBoundsX(xUB);
			layer.setUpperBoundsS(sUB);
//			System.out.println(k + " " + Arrays.toString(xUB));
//			System.out.println(k + " " + Arrays.toString(sUB));
		}
	}
	
	/**
	 * Method used to read and set the weights from a file
	 * @param filename		Name of the file containing the weights
	 * @throws FileNotFoundException
	 */
	private void readWeights(String filename) throws FileNotFoundException {
		// open the file 
		Scanner s = new Scanner(new File(filename));
		// use commas as delimiters
		s.useDelimiter(",");
		
		// get the number of pixels in the input and create the input layer
		int n_0 = s.nextInt();
		this.layers[0] = new Layer(0, n_0, null, null);
		
		// set the bounds on the input 
		double[] x_0UB = new double[n_0];
		Arrays.fill(x_0UB, 1);
		this.layers[0].setUpperBoundsX(x_0UB);
		s.close();
		
		// open the file again
		s = new Scanner(new File(filename));
		
		// create arrays for the weights and biases
		double[][] weights = null;
		double[] bias = null;
		
		// use a counter to see if the next array contains weights or biases
		int h = 0;
		
		while (s.hasNext()) {
			// get the dimensions of the next array
			String[] line = s.nextLine().split(",");
			int nRows = Integer.parseInt(line[0]);
			int nCols = Integer.parseInt(line[1]);
			
			// check if the array contains weights or biases
			if (h % 2 == 0) {
				weights = new double[nCols][nRows];
			}
			else {
				bias = new double[nRows];
			}
			
			// read the weights/biases
			for (int y=0; y < nRows; y++) {
				String[] row = s.nextLine().split(",");
				for (int x=0; x < nCols; x++) {
					if (h % 2 == 0) {
						weights[x][y] = Double.parseDouble(row[x]);
					}
					else {
						bias[y] = Double.parseDouble(row[x]);
					}
				}
				
			}
		
			// if the read array contained biases set the weights and biases for the corresponding layer
			if (h % 2 != 0) {
				this.layers[(h / 2) + 1] = new Layer((h / 2) + 1, bias.length, weights, bias);
			}
			// increment the counter
			h++;
		}
		// close the scanner
		s.close();
	}
	
}
