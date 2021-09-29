import java.io.IOException;
import java.util.Arrays;

import ilog.concert.IloException;


/**
 * Class used to create a feature visualization
 * @author Tim Tjhay (495230tt)
 */
public class MainVisualize {

	/**
	 * Main method used to create the feature visualization
	 * @param args
	 * @throws IloException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IloException, IOException {
		// create an array representing the DNN the visualization is made for
		int[] architecture = {8,8,8}; 

		// get the name of the file containing the weights
		String weightsFile = "input//weights//";
		String archString = "";
		for (int n_k: architecture) {
			archString += n_k + "_";
		}
		archString = archString.substring(0, archString.length()-1);
		weightsFile += archString + "//weights.csv";

		// create the DNN
		DNN dnn = new DNN(weightsFile, architecture.length + 1);

		// for every digit:
		for (int i=0; i < 10; i++) {
			// initialize the model used to create the visualization of the digit i
			MILPVisualize modelVisualize = new MILPVisualize(dnn, i);
			// create the visualization and print the runtime
			long startTime = System.currentTimeMillis();
			modelVisualize.solve();
			double runTime = (double) (System.currentTimeMillis() - startTime) / 1000;
			System.out.println(Arrays.toString(architecture) + " " + i + "/9: " + runTime);
			
			// write the visualization to a file
			modelVisualize.createVisualization("output//featureVisualization//visualize" + i + ".csv");
			
			// clean up the model
			modelVisualize.cleanup();
		}
	}

}
