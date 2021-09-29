import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import ilog.concert.IloException;


/**
 * Class used to test the performance of the bound tightening method 
 * proposed by Fischetti and Jo (2018) by creating adversarial examples
 * @author Tim Tjhay (495230tt)
 */
public class Main {

	/**
	 * Main method that creates the adversarial examples for the five DNNs
	 * and writes the performance statistics into a csv file
	 * @param args
	 * @throws IloException
	 * @throws IOException
	 * @throws CloneNotSupportedException
	 */
	public static void main(String[] args) throws IloException, IOException, CloneNotSupportedException {

		// create array of architectures of DNNs that the test needs to be run for (all, 1-4, 5)
//		int[][] architectures = {{8,8,8}, {8,8,8,8,8}, {20,10,8,8}, {20,10,8,8,8}, {20,20,10,10,10}}; 
		int[][] architectures = {{8,8,8}, {8,8,8,8,8}, {20,10,8,8}, {20,10,8,8,8}}; 
//		int[][] architectures = {{20,20,10,10,10}};
		
		// choose if optimalityGap should be used during test
		boolean optimalityGap = false;
		
		// open a writer to write the performance statistics and choose filename based on architectures and optimalityGap
		BufferedWriter w = new BufferedWriter(new FileWriter("output//solveData//solveDataIndicator14.csv"));
		
		// keep track of how many architectures have been tested
		int arch = 1;
		int nArchs = architectures.length;

		// for all DNNs:
		for (int[] architecture: architectures) {
			// get the name of the file containing the weights
			String weightsFile = "input//weights//";
			String archString = "";
			for (int n_k: architecture) {
				archString += n_k + "_";
			}
			archString = archString.substring(0, archString.length()-1);
			weightsFile += archString + "//weights.csv";
			
			// initialize the DNN using the weights
			DNN dnn = new DNN(weightsFile, architecture.length + 1);

			// get the original images and their classifications
			String testdata = "input//testdata//" + archString;
			double[][] images = readImages(testdata + "//images.csv");
			int[] digits = readClass(testdata + "//classifications.csv");

			// create an array of the possible time limits on the bound tightening
			// corresponding to the three models with the first true corresponding to no bound tightening
			// (base, improved, weaker improved)
			boolean[] timeLimitBounds = {true, true, false};
			
			// write which architecture the performance statistics belong to
			w.write(arch + ",");
			// for every one of the three models:
			for (int h=0; h < timeLimitBounds.length; h++) {
				// initialize the time needed to tighten the bounds
				double presolveTime = 0;
				
				if (h > 0) {
					// tighten the bounds depending on the used model and 
					// keep track of the time needed to tighten these bounds
					long startPresolve = System.currentTimeMillis();
					dnn.calculateBounds(timeLimitBounds[h]);
					// convert the time from milliseconds to seconds
					presolveTime = (double) (System.currentTimeMillis() - startPresolve)/ 1000;
				}

				// initialize performance measures
				int nSolved = 0;
				double totalGap = 0;
				double totalTime = 0;
				double aveNodes = 0;
				double totalObj = 0;
				
				// set the maximum deviation used while creating the adversarial examples
				double maxDeviation = 1; 

				// for all images:
				for (int i=0; i < images.length; i++) {
					// get the image and calculate its target class using its 
					// original classification
					double[] input = images[i];
					int targetDigit = (digits[i] + 5) % 10;
					
					// print which adversarial example is currently being created
					// to make it possible to track progress
					System.out.println(arch + "/" + nArchs + ": " + (i+1) + "/100	(" + (h+1) + "/3)	" + digits[i] + " to " + targetDigit + "	");

					// initialize the MILP model 
					MILPAdversarial advExmplModel = new MILPAdversarial(dnn, input, targetDigit, maxDeviation, optimalityGap); 
					
					// create the adversarial example by solving the MILP and keep track of time needed
					long start = System.currentTimeMillis();
					// store if MILP was optimally solved
					boolean solved = advExmplModel.solve();
					long time = System.currentTimeMillis()-start;
					
					// update performance measures
					totalTime += (double) time / 1000;
					aveNodes += (double) advExmplModel.getNodes()/100;
					totalGap += advExmplModel.getGap();

					if (solved) {
						nSolved++;
						totalObj += advExmplModel.getObj();
					}
					
//					// write adversarial example to a file so an image can be made of it
//					advExmplModel.createAdvExmpl("output//examples//adversarial//csvMaxDev2//" + digits[i] + "to" + targetDigit + ".csv", "output//examples//original//csv//" + digits[i] + "to" + targetDigit + ".csv");
					// cleanup the model
					advExmplModel.cleanup();
				}
				
				// get which model was used
				String model = "";
				if (h == 0) {
					model = "base";
				}
				else if (h == 1) {
					model = "weak";
				}
				else {
					model = "improved";
				}
				
				// write the model name and the statistics to the file and print it as well
				w.write(model + "," +  nSolved + "," + totalGap + "," + aveNodes + "," + presolveTime + "," + (double) totalTime/100 + "," + (double) totalObj/nSolved + ",");
				System.out.println(model + "," +  nSolved + "," + totalGap + "," + aveNodes + "," + presolveTime + "," + (double) totalTime/100 + "," + (double) totalObj/nSolved);
			}
			// go to the next line in the file for the next DNN
			w.write("\n ");
			arch++;
		}
		// close the writer
		w.close();
	}

	/**
	 * Method that reads a set of images from a file into a 2d-array
	 * @param imageFilename		Name of the file containing the images
	 * @return					2d double array containing the images
	 * @throws FileNotFoundException
	 */
	public static double[][] readImages(String imageFilename) throws FileNotFoundException {
		// open the file
		Scanner sImage = new Scanner(new File(imageFilename));

		// get the number of images in the file
		int n = Integer.parseInt(sImage.nextLine());

		// create the 2d-array that will store the images
		double[][] images = new double[n][];

		// read the images
		int j = 0;
		while (sImage.hasNext()) {
			String[] imageStr = sImage.nextLine().split(",");
			double[] image = new double[28*28];
			// convert array of strings to array of doubles
			for (int i=0; i < image.length ; i++) {
				image[i] = Double.parseDouble(imageStr[i]);
			}
			images[j] = image;
			j++;
		}

		return images;
	}

	/**
	 * Method that reads a set of classifications from a file into an array
	 * @param classFilename		Name of the file
	 * @return					1d int array containing the integer classifications
	 * @throws FileNotFoundException
	 */
	public static int[] readClass(String classFilename) throws FileNotFoundException {
		// open the file
		Scanner sClass = new Scanner(new File(classFilename));

		// get the number of classifications in the file
		int n = Integer.parseInt(sClass.nextLine());
		
		// create the array that will store the classifications
		int[] classifications = new int[n];

		// read the classifications
		int j = 0;
		while (sClass.hasNext()) {
			int classification = Integer.parseInt(sClass.nextLine());
			classifications[j] = classification;
			j++;
		}

		return classifications;
	}
	
	/**
	 * Method that reads a set of 450 adversarial examples from a file into a 2d-array
	 * @param 	imageFilename	Name of file containing adversarial examples
	 * @return					2d double array containing adversarial examples
	 * @throws FileNotFoundException
	 */
	public static double[][] readAdvExmpls(String imageFilename) throws FileNotFoundException {
		// open the file
		Scanner sAdvExmpls = new Scanner(new File(imageFilename));

		// create the 2d-array
		double[][] advExmpls = new double[450][];

		// read the adversarial examples
		int j = 0;
		while (sAdvExmpls.hasNext()) {
			String[] advExmplStr = sAdvExmpls.nextLine().split(",");
			double[] advExmpl = new double[28*28];
			for (int i=0; i < advExmpl.length ; i++) {
				advExmpl[i] = Double.parseDouble(advExmplStr[i]);
			}
			advExmpls[j] = advExmpl;
			j++;
		}


		return advExmpls;
	}

	/**
	 * Method that reads the correct classification for a set of 450 adversarial examples
	 * from a file into an array
	 * @param classFilename		Name of the file
	 * @return					1d int array containing the correct classifications
	 * @throws FileNotFoundException
	 */
	public static int[] readAdvExmplsClass(String classFilename) throws FileNotFoundException {
		// open the file
		Scanner sClass = new Scanner(new File(classFilename));

		// create the array
		int[] classifications = new int[450];

		// read the correct classifications
		int j = 0;
		while (sClass.hasNext()) {
			int classification = Integer.parseInt(sClass.nextLine());
			classifications[j] = classification;
			j++;
		}


		return classifications;
	}

}
