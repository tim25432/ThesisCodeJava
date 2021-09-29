import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import ilog.concert.IloException;
import ilog.cplex.IloCplex.UnknownObjectException;


/**
 * Class used to create a data set of adversarial examples
 * @author Tim Tjhay (495230tt)
 */
public class MainAdvExmplSet {

	/**
	 * Main method used to create training and test data sets of adversarial examples 
	 * @param args
	 * @throws UnknownObjectException
	 * @throws IloException
	 * @throws IOException
	 * @throws CloneNotSupportedException
	 */
	public static void main(String[] args) throws UnknownObjectException, IloException, IOException, CloneNotSupportedException {
		// create an array of DNNs to create adversarial examples for
//		int[][] architectures = {{8,8,8}, {8,8,8,8,8}, {20,10,8,8}, {20,10,8,8,8}, {20,20,10,10,10}}; 
		int[][] architectures = {{8,8,8}}; 

		// for every DNN:
		for (int[] architecture: architectures) {
			// get the name of the file containing the weights
			String weightsFile = "input//weights//";
			String archString = "";
			for (int n_k: architecture) {
				archString += n_k + "_";
			}
			archString = archString.substring(0, archString.length()-1);
			weightsFile += archString + "//weights.csv";

			// initialize the DNN and apply the bound tightening method without a time limit
			DNN dnn = new DNN(weightsFile, architecture.length + 1);
			dnn.calculateBounds(false);
			
			// open writers for the training data sets adversarial examples and their corresponding correct classification
			BufferedWriter wAdvExmplTrain = new BufferedWriter(new FileWriter("output//advExmpls//" + archString + "//images.csv"));
			BufferedWriter wDigitsTrain = new BufferedWriter(new FileWriter("output//advExmpls//" + archString + "//classifications.csv"));

			// read the images and classifications the adversarial examples will be created from
			String testdata = "input//testdata//" + archString;
			double[][] imagesTrain = Main.readImages(testdata + "//imagesOrdered.csv");
			int[] digitsTrain = Main.readClass(testdata + "//classificationsOrdered.csv");
			
			// write the adversarial examples
			writeAdvExmpls(dnn, imagesTrain, digitsTrain, wAdvExmplTrain, wDigitsTrain);
			
			// close the writers
			wAdvExmplTrain.close();
			wDigitsTrain.close();

			// do the same for the test data set
			BufferedWriter wAdvExmplTest = new BufferedWriter(new FileWriter("output//advExmpls//" + archString + "//imagesTest.csv"));
			BufferedWriter wDigitsTest = new BufferedWriter(new FileWriter("output//advExmpls//" + archString + "//classificationsTest.csv"));
			
			double[][] imagesTest = Main.readImages(testdata + "//imagesOrdered2.csv");
			int[] digitsTest = Main.readClass(testdata + "//classificationsOrdered2.csv");

			writeAdvExmpls(dnn, imagesTest, digitsTest, wAdvExmplTest, wDigitsTest);
			
			wAdvExmplTest.close();
			wDigitsTest.close();
		}
	}
	
	/**
	 * Method used to write adversarial examples and their corresponding correct classifcation to files
	 * @param dnn			The DNN used
	 * @param images		The original images
	 * @param digits		The classification of the images
	 * @param wAdvExmpl		A BufferedWriter for the adversarial examples
	 * @param wDigits		A BufferedWriter for the correct classifications
	 * @throws IOException
	 * @throws IloException
	 */
	public static void writeAdvExmpls(DNN dnn, double[][] images, int[] digits, BufferedWriter wAdvExmpl, BufferedWriter wDigits) throws IOException, IloException {
		// set the maximum deviation
		double maxDeviation = 1; 

		// for every original image:
		for (int i=0; i < images.length; i++) {
			// get the original image
			double[] input = images[i];
			// for every digit that is not the correct classification:
			for (int targetDigit=0; targetDigit < 10; targetDigit++) {
				if (targetDigit != digits[i]) {
					// print the progress
					System.out.println((i+1) + "/" + images.length + "	" + digits[i] + " to " + targetDigit + "	");

					// initialize the model to create the adversarial example
					MILPAdversarial advExmplModel = new MILPAdversarial(dnn, input, targetDigit, maxDeviation, false);

					// create the adversarial example
					boolean solved = advExmplModel.solve();

					// if the model was optimally solved write the resulting adversarial example and its correct classification to the files
					if (solved) {
						advExmplModel.writeAdvExmpl(wAdvExmpl);
						wDigits.write(digits[i] + "\n");
					}
					// clean up the model
					advExmplModel.cleanup();
				}
			}
		}
	}

}
