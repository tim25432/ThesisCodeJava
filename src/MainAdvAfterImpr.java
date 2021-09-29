import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import ilog.concert.IloException;


/**
 * Class used to test the performance of the proposed improvement methods 
 * by creating new adversarial examples using these methods
 * @author Tim Tjhay (495230tt)
 */
public class MainAdvAfterImpr {

	/**
	 * Main method used to create the new adversarial examples after applying the accuracy improvement methods
	 * and write the performance to csv files
	 * @param args
	 * @throws IloException
	 * @throws IOException
	 * @throws CloneNotSupportedException
	 */
	public static void main(String[] args) throws IloException, IOException, CloneNotSupportedException {
		// create an array with the names of the files containing the (retrained) weights
		String[] weightFiles = {"//weightsIP.csv", "//weightsPR.csv", "//weightsCR.csv"};

		// open a writer
		BufferedWriter w = new BufferedWriter(new FileWriter("output//solveData//solveDataAfterImpr.csv"));

		// create a string containing the directory of the weight files
		String directory = "input//weights//afterImpr";

		// get the original images and classifications
		String testdata = "input//testdata//afterImpr";
		double[][] images = Main.readImages(testdata + "//images.csv");
		int[] digits = Main.readClass(testdata + "//classifications.csv");

		// get the perturbation
		double[][] perturbation = readPerturbation("output//perturbation//8_8_8//perturbation1//perturbationMinDist.csv");

		// for every improvement method:
		for (String weightFile: weightFiles) {
			// create a DNN using the weights file and apply the bound tightening
			DNN dnn = new DNN(directory + weightFile, 4);
			dnn.calculateBounds(false);

			// store which is used
			String approach = weightFile.substring(9, 11);
			
			if (weightFile.contains("IP")) {
				// if the method is the perturbation method also write the performance 
				// of the base model as the weights are the same
				writePerformance(w, dnn, images, digits, false, perturbation, "base");
				
				// for the perturbation widen the input interval as an input
				// over 1 is possible after perturbation and tighten bounds again
				Layer inputLayer = dnn.getLayers()[0];
				double[] x_0UB = new double[inputLayer.getN()];
				Arrays.fill(x_0UB, 1.5);
				dnn.getLayers()[0].setUpperBoundsX(x_0UB);
				dnn.calculateBounds(false);
				
				// write the performance
				writePerformance(w, dnn, images, digits, true, perturbation, approach);
			}
			else {
				writePerformance(w, dnn, images, digits, false, perturbation, approach);
			}
		}
		// close the writer
		w.close();
	}
	
	/**
	 * Method used to write several performance measures of the creation of adversarial examples
	 * after applying an accuracy improvement method
	 * @param w					The BufferedWriter used to write the performance measures
	 * @param dnn				The used DNN
	 * @param images			The original images
	 * @param digits			The classifications of the images
	 * @param perturb			If the input should be perturbed
	 * @param perturbation		The perturbation
	 * @param approach			The used improvement approach
	 * @throws IloException		
	 * @throws IOException		
	 */
	public static void writePerformance(BufferedWriter w, DNN dnn, double[][] images, int[] digits, boolean perturb, double[][] perturbation, String approach) throws IloException, IOException {
		// initialize the statistics
		int nSolved = 0;
		double totalGap = 0;
		double totalTime = 0;
		double aveNodes = 0;
		double totalObj = 0;

		// for all original images:
		for (int i=0; i < 100; i++) {
			// get the image and the target classification
			double[] input = images[i];
			int targetDigit = (digits[i] + 5) % 10;
			// print the progress
			System.out.println(approach + ": " + (i+1) + "/100	" + digits[i] + " to " + targetDigit + "	");

			// initialize the model
			MILPAdvAfterImpr advExmplModel = new MILPAdvAfterImpr(dnn, input, targetDigit, perturb, perturbation); 
			// create the adversarial example and keep track of the run time
			long start = System.currentTimeMillis();
			boolean solved = advExmplModel.solve();
			long time = System.currentTimeMillis()-start;
			
			// update the statistics
			totalTime += (double) time / 1000;
			aveNodes += (double) advExmplModel.getNodes()/100;
			totalGap += advExmplModel.getGap();

			if (solved) {
				nSolved++;
				totalObj += advExmplModel.getObj();
			}
			// clean up the model
			advExmplModel.cleanup();
		}

		// write the statistics to the file and print them
		w.write(approach + "," +  nSolved + "," + totalGap + "," + aveNodes + "," + (double) totalTime/100 + "," + (double) totalObj/nSolved + ",");
		System.out.println(approach + "," + nSolved + "," + totalGap + "," + aveNodes + "," + (double) totalTime/100 + "," + (double) totalObj/nSolved);
	}

	/**
	 * Method used to read a perturbation from a file into a 2d-array
	 * @param perturbationFilename		Name of the file containing the perturbation
	 * @return							2d double array containing the perturbation with the first row 
	 * 									corresponding to the weights and the second to the disturbances
	 * @throws FileNotFoundException
	 */
	public static double[][] readPerturbation(String perturbationFilename) throws FileNotFoundException {
		// open the file
		Scanner sPerturb = new Scanner(new File(perturbationFilename));

		// create a 2d-array for the perturbation
		double[][] perturbation = new double[2][28*28];

		// read the perturbation
		for (int i=0; i < 28*28; i++) {
			String[] perturbStr = sPerturb.nextLine().split(",");

			perturbation[0][i] = Double.parseDouble(perturbStr[0]);
			perturbation[1][i] = Double.parseDouble(perturbStr[1]);
		}

		return perturbation;
	}

}
