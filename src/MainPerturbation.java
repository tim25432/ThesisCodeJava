import java.io.IOException;
import java.util.Arrays;

import ilog.concert.IloException;


/**
 * Class used to create the perturbation
 * @author Tim Tjhay (495230tt)
 */
public class MainPerturbation {

	/**
	 * Main method used to create the perturbation and write it to a csv file
	 * @param args
	 * @throws IloException
	 * @throws CloneNotSupportedException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IloException, CloneNotSupportedException, IOException {
		// create an array containing the DNNs the perturbation should be created for
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

			// initialize the DNN
			DNN dnn = new DNN(weightsFile, architecture.length + 1);
			
			// set a the lower bound on the input layer to -1
			double[] x_0LB = new double[dnn.getLayers()[0].getN()];
			Arrays.fill(x_0LB, -1);
			dnn.getLayers()[0].setLowerBoundsX(x_0LB);
			
			// tighten the bounds
			dnn.calculateBounds(false);

			// get the training data(original images + adversarial examples)
			String traindata = "input//testdata//" + archString;
			String adversarial = "output//advExmpls//" + archString;

			double[][] images = Main.readImages(traindata + "//imagesOrdered.csv");
			int[] classification = Main.readClass(traindata + "//classificationsOrdered.csv");
			double[][] advExmpls = Main.readAdvExmpls(adversarial + "//images.csv");
			int[] advExmplsClass = Main.readAdvExmplsClass(adversarial + "//classifications.csv");
			
			//create perturbations
			// should perturbation 1 be created as the minimum distance variation
			boolean minDist = true;
			createPerturb1(minDist, dnn, images, classification, advExmpls, advExmplsClass);
//			createPerturb2(dnn, images, classification, advExmpls, advExmplsClass);
//			createPerturbOnlyWeights(dnn, images, classification, advExmpls, advExmplsClass);
//			createPerturbOnlyDisturbances(dnn, images, classification, advExmpls, advExmplsClass);
		}
	}

	/**
	 * Method used to create perturbation 1
	 * @param minDist				Should minimum distance variation be created
	 * @param dnn					The used DNN
	 * @param images				The original images
	 * @param classification		The classifications of the images
	 * @param advExmpls				The adversarial examples
	 * @param advExmplsClass		The correct classification of the adv. examples
	 * @throws IloException
	 * @throws IOException
	 */
	public static void createPerturb1(boolean minDist, DNN dnn, double[][] images, int[] classification, double[][] advExmpls, int[] advExmplsClass) throws IloException, IOException {
		// create arrays for the training data
		double[][] trainSet = new double[30][];
		int[] correctClass = new int[30];

		// initialize a counter to store the index of the training data
		int i = 0;

		// add an original image of every digit
		for (int j=0; j < images.length; j++) {
			if (j % 5 < 1) {
				trainSet[i] = images[j];
				correctClass[i] = classification[j];
				i++;
			}
		}

		// add 2 adversarial examples of every digit(as correct classification)
		// the two following digits are chosen as target digit(e.g. 0->1 and 0->2 were added)
		for (int j=0; j < images.length; j++) {
			if (j % 5 < 1) {
				int digit = advExmplsClass[9*j];
				for (int h=0; h < 2; h++) {
					trainSet[i] = advExmpls[9*j + digit + h];
					correctClass[i] = advExmplsClass[9*j + digit + h];
					i++;
				}
			}
		}

		// initialize the model used to create the perturbation
		MILPPerturbation perturbModel = new MILPPerturbation(dnn, trainSet, correctClass, true, true, minDist);
		
		// create the perturbation while keeping track of the running time
		long startTime = System.currentTimeMillis();
		perturbModel.solve();
		// print the running time in seconds
		System.out.println("time: " + ((double) (System.currentTimeMillis() - startTime)/1000));
		
		// write the perturbation to a file
		String filename = "output//perturbation//8_8_8//perturbation1//perturbation";
		if (minDist) {
			filename += "MinDist";
		}
		filename += ".csv";
		perturbModel.writePQ(filename);
		
		// clean up the model
		perturbModel.cleanup();
	}
	
	/**
	 * Method used to create perturbation 1 using only the weights
	 * @param dnn				The used DNN
	 * @param images			The original images
	 * @param classification	The classification of the images
	 * @param advExmpls			The adversarial examples
	 * @param advExmplsClass	The correct classification of the adversarial examples
	 * @throws IloException
	 * @throws IOException
	 */
	public static void createPerturbOnlyWeights(DNN dnn, double[][] images, int[] classification, double[][] advExmpls, int[] advExmplsClass) throws IloException, IOException {
		// create arrays for training data
		double[][] trainSet = new double[30][];
		int[] correctClass = new int[30];

		// add training data in the same way as for perturbation 1
		int i = 0;

		for (int j=0; j < images.length; j++) {
			if (j % 5 < 1) {
				trainSet[i] = images[j];
				correctClass[i] = classification[j];
				i++;
			}
		}

		for (int j=0; j < images.length; j++) {
			if (j % 5 < 1) {
				int digit = advExmplsClass[9*j];
				for (int h=0; h < 2; h++) {
					trainSet[i] = advExmpls[9*j + digit + h];
					correctClass[i] = advExmplsClass[9*j + digit + h];
					i++;
				}
			}
		}

		// create the perturbation using only weights and print the running time
		System.out.println("only weights");
		MILPPerturbation perturbModel = new MILPPerturbation(dnn, trainSet, correctClass, true, false, false);
		long s = System.currentTimeMillis();
		perturbModel.solve();
		System.out.println("time: " + ((double) (System.currentTimeMillis() - s)/1000));
		
		// write the perturbation to a file
		perturbModel.writePQ("output//perturbation//8_8_8//perturbation1//perturbationWeights.csv");
		
		// clean up the model
		perturbModel.cleanup();
	}

	/**
	 * Method used to create perturbation 1 using only the disturbances
	 * @param dnn				The used DNN
	 * @param images			The original images
	 * @param classification	The classification of the images
	 * @param advExmpls			The adversarial examples
	 * @param advExmplsClass	The correct classification of the adversarial examples
	 * @throws IloException
	 * @throws IOException
	 */
	public static void createPerturbOnlyDisturbances(DNN dnn, double[][] images, int[] classification, double[][] advExmpls, int[] advExmplsClass) throws IloException, IOException {
		// create an array for the training data
		double[][] trainSet = new double[10][];
		int[] correctClass = new int[10];

		int i = 0;

		// add an adversarial example of every digit
		for (int j=0; j < images.length; j++) {
			if (j % 5 < 1) {
				int digit = advExmplsClass[9*j];
				for (int h=0; h < 1; h++) {
					trainSet[i] = advExmpls[9*j + digit + h];
					correctClass[i] = advExmplsClass[9*j + digit + h];
					i++;
				}
			}
		}
		
		// create the perturbation using only disturbances and print the running time
		System.out.println("only disturbances");
		MILPPerturbation perturbModel = new MILPPerturbation(dnn, trainSet, correctClass, false, true, false);
		long s = System.currentTimeMillis();
		perturbModel.solve();
		System.out.println("time: " + ((double) (System.currentTimeMillis() - s)/1000));
		
		// write the perturbation to a file
		perturbModel.writePQ("output//perturbation//8_8_8//perturbation1//perturbationDisturbances.csv");
		
		// clean up the model
		perturbModel.cleanup();
	}

	/**
	 * Method used to create perturbations 2.c , c=0,...,9
	 * @param dnn				The used DNN
	 * @param images			The original images
	 * @param classification	The classification of the images
	 * @param advExmpls			The adversarial examples
	 * @param advExmplsClass	The correct classification of the adversarial examples
	 * @throws IloException
	 * @throws IOException
	 */
	public static void createPerturb2(DNN dnn, double[][] images, int[] classification, double[][] advExmpls, int[] advExmplsClass) throws IloException, IOException {
		// create array for the training data
		double[][] trainSet = new double[23][];
		int[] correctClass = new int[23];

		// for every digit:
		for (int targetDigit=0; targetDigit < 10; targetDigit++) {
			// print the digit
			System.out.println(targetDigit);

			int i = 0;

			// add 5 original images of the digit to the training data
			for (i=0; i < 5; i++) {
				trainSet[i] = images[5*targetDigit+i];
				correctClass[i] = classification[5*targetDigit+i];
			}

			// add 2 adversarial examples of every other digit(as the source) to
			// the training data that have the current digit as their target classification
			for (int j=0; j < images.length; j++) {
				if (j % 5 < 2 && classification[j] != targetDigit) {
					if (classification[j] > targetDigit) {
						trainSet[i] = advExmpls[9*j + targetDigit];
						correctClass[i] = advExmplsClass[9*j + targetDigit];
					}
					else {
						trainSet[i] = advExmpls[9*j + targetDigit - 1];
						correctClass[i] = advExmplsClass[9*j + targetDigit - 1];
					}
					i++;
				}
			}

			// create the perturbation and print the run time
			MILPPerturbation perturbModel = new MILPPerturbation(dnn, trainSet, correctClass, true, true, false);
			long s = System.currentTimeMillis();
			perturbModel.solve();
			System.out.println("time: " + ((double) (System.currentTimeMillis() - s)/1000));
			
			// write the perturbation to a file
			perturbModel.writePQ("output//perturbation//8_8_8//perturbation2//perturbations//perturbation" + targetDigit + ".csv");
			
			// clean up the model
			perturbModel.cleanup();
		}
	}

}
