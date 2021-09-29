import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.UnknownObjectException;


/**
 * Class used to model the MILP formulation used to create adversarial examples
 * @author Tim Tjhay (495230tt)
 */
public class MILPAdversarial {
	private IloCplex cplex;
	
	private DNN dnn;
	private double[] input;
	private int targetDigit;
	private double maxDeviation;
	
	private IloNumVar[] dVarList;
	private Map<Layer,IloNumVar[]> xVarMap;
	private Map<Layer,IloNumVar[]> sVarMap;
	private Map<Layer,IloNumVar[]> zVarMap;
	
	private IloConstraint[] targetConstr;
	private IloConstraint[] defDConstr;
	
	

	/**
	 * Initialization of MILP model used to create adversarial examples
	 * @param dnn				The DNN that adversarial examples need to be created for
	 * @param input				The original image
	 * @param targetDigit		The digit that the adversarial example needs to be classified as
	 * @param maxDeviation		The maximum deviation from the original image
	 * @param gapTolerance		If the model should be solved using a 1% optimality gap
	 * @throws IloException		
	 */
	public MILPAdversarial(DNN dnn, double[] input, int targetDigit, double maxDeviation, boolean gapTolerance) throws IloException {
		this.cplex = new IloCplex();
		// stop cplex from printing the output
		this.cplex.setOut(null);
		
		// store the relevant data
		this.dnn = dnn;
		this.input = input;
		this.targetDigit = targetDigit;
		this.maxDeviation = maxDeviation;
		
		// create maps to store the variables
		this.dVarList = new IloNumVar[input.length];
		this.xVarMap = new HashMap<>();
		this.sVarMap = new HashMap<>();
		this.zVarMap = new HashMap<>();
		
		// create the variables and add the objective
		createVariables();
		addObjective();
		
		// add the constraints
		addDefNeuron();
		addXSZConstraints();
		addTargetClass();
		addDefD();
		addMaxDeviation();
		
		// set a time limit of 300 seconds
		this.cplex.setParam(IloCplex.Param.TimeLimit, 300);
		if (gapTolerance) {
			// allow an optimality gap of 1%
			this.cplex.setParam(IloCplex.Param.MIP.Tolerances.MIPGap, 0.01);
		}
	}
	
	/**
	 * Method used to create the variables
	 * @throws IloException
	 */
	private void createVariables() throws IloException {
		// for all layers of the DNN
		for (Layer k: this.dnn.getLayers()) {
			// create lists for the x and s variables in this layer
			this.xVarMap.put(k, new IloNumVar[k.getN()]);
			this.sVarMap.put(k, new IloNumVar[k.getN()]);
			if (k != this.dnn.getLayers()[0]) {
				// create a list for the z variable if this layer is not the input layer
				this.zVarMap.put(k, new IloNumVar[k.getN()]);
			}
			
			// get the bounds of x and s
			double[] xLB = k.getLowerBoundsX();
			double[] xUB = k.getUpperBoundsX();
			double[] sLB = k.getLowerBoundsS();
			double[] sUB = k.getUpperBoundsS();
			
			// add the variables
			for (int i=0; i < k.getN(); i++) {
				this.xVarMap.get(k)[i] = this.cplex.numVar(xLB[i], xUB[i]);
				this.sVarMap.get(k)[i] = this.cplex.numVar(sLB[i], sUB[i]);
				if (k != this.dnn.getLayers()[0]) {
					this.zVarMap.get(k)[i] = this.cplex.boolVar();
				}
			}
		}
		
		// create the disturbance variable for every pixel
		for (int i=0; i < this.dVarList.length; i++) {
			this.dVarList[i] = this.cplex.numVar(0, Integer.MAX_VALUE);
		}
	}
	
	/**
	 * Method used to add the objective function
	 * @throws IloException
	 */
	private void addObjective() throws IloException {
		// make minimizing the total disturbance the objective
		IloNumExpr obj = this.cplex.sum(this.dVarList);
		this.cplex.addMinimize(obj);
	}
	
	/**
	 * Method used to add the constraint that imposes the 
	 * definition of a neuron
	 * @throws IloException
	 */
	private void addDefNeuron() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		// for all layers except the input layer:
		for (int k=1; k < layers.length; k++) {
			// get the weights and biases for this layer
			double[][] w = layers[k].getWeights();
			double[] b = layers[k].getBias();
			
			// get the x and s variables of this layer
			IloNumVar[] x_k = this.xVarMap.get(layers[k]);
			IloNumVar[] s_k = this.sVarMap.get(layers[k]);

			// get the x variables of the previous layer
			IloNumVar[] x_k_1 = this.xVarMap.get(layers[k-1]);
			
			// for every neuron in this layer:
			for (int j=0; j < x_k.length; j++) {
				// set the right hand side
				IloNumExpr rhs = this.cplex.diff(x_k[j], s_k[j]);
				// initialize the left hand side as the bias of this neuron
				IloNumExpr lhs = this.cplex.constant(b[j]);
				
				// add the sum over i of the w_k_i*x_k-1_i
				for (int i=0; i < x_k_1.length; i++) {
					IloNumExpr wx = this.cplex.prod(w[j][i], x_k_1[i]);
					lhs = this.cplex.sum(lhs, wx);
				}
				
				// add the constraint
				this.cplex.addEq(lhs, rhs);
			}
		}
	}
	
	/**
	 * Method used to add the indicator constraints that use z
	 * @throws IloException
	 */
	private void addXSZConstraints() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		// for all layers except the input layer:
		for (int k=1; k < layers.length; k++) {
			// get the x, s and z variables of this layer
			IloNumVar[] x = this.xVarMap.get(layers[k]);
			IloNumVar[] s = this.sVarMap.get(layers[k]);
			IloNumVar[] z = this.zVarMap.get(layers[k]);
			// for all neurons in the layer:
			for (int j=0; j < x.length; j++) {
				// add the constraint that x equals 0 if z equals 1
				IloConstraint zConstr1 = this.cplex.eq(z[j], 1);
				IloConstraint xConstr = this.cplex.eq(x[j], 0);
				IloConstraint xzConstr = this.cplex.ifThen(zConstr1, xConstr);
				this.cplex.add(xzConstr);
				
				// add the constraint that s equals 0 if z equals 0
				IloConstraint zConstr0 = this.cplex.eq(z[j], 0);
				IloConstraint sConstr = this.cplex.eq(s[j], 0);
				IloConstraint szConstr = this.cplex.ifThen(zConstr0, sConstr);
				this.cplex.add(szConstr);
			}
		}
	}
	
	/**
	 * Method used to add the constraint that sets target classification
	 * @throws IloException
	 */
	private void addTargetClass() throws IloException {
		// get the x variables that correspond to the output layer
		Layer[] layers = this.dnn.getLayers(); 
		Layer outputLayer = layers[layers.length - 1];
		IloNumVar[] x_K = this.xVarMap.get(outputLayer);
		
		// set the left hand side
		IloNumExpr lhs = x_K[this.targetDigit];

		// store the constraints that will be added
		this.targetConstr = new IloConstraint[x_K.length-1];
		int i = 0;
		
//		// add a lower bound on the activation of the target classification
//		this.targetConstr = new IloConstraint[x_K.length];
//		this.targetConstr[0] = this.cplex.addGe(lhs, this.cplex.constant(0.01));
//		int i = 1;
		
		// add constraints that impose the activation of the target classifications neuron
		// is at least 20% greater than activation of the other neurons
		for (int j=0; j < x_K.length; j++) {
			if (j != this.targetDigit) {
				IloNumExpr rhs = this.cplex.prod(1.2, x_K[j]);
				
				this.targetConstr[i] = this.cplex.addGe(lhs, rhs);
				i++;
			}
		}
	}

	/**
	 * Method used to add constraint that defines the disturbance d
	 * @throws IloException
	 */
	private void addDefD() throws IloException {
		// get the x variables corresponding to the input layer
		Layer inputLayer  = this.dnn.getLayers()[0];
		IloNumVar[] x_0 = this.xVarMap.get(inputLayer);
		
		// store the constraints that will be added
		this.defDConstr = new IloConstraint[this.dVarList.length*2];
		
		// for every pixel of the input image
		for (int j=0; j < this.dVarList.length; j++) {
			// set the left and right hand side of the constraint
			IloNumExpr lhs = this.dVarList[j];
			IloNumExpr rhs = this.cplex.diff(x_0[j], this.input[j]);
			
			// add the constraint
			this.defDConstr[2*j] = this.cplex.addGe(lhs, rhs);
			
			// multiply the left hand side by -1 to allow negative disturbances 
			lhs = this.cplex.prod(this.cplex.constant(-1), lhs);
			this.defDConstr[2*j + 1] =  this.cplex.addLe(lhs, rhs);
		}
	}
	
	/**
	 * Method used to add the constraint that imposes the maximum deviation
	 * @throws IloException
	 */
	private void addMaxDeviation() throws IloException {
		// make the maximum deviation an upper bound for every disturbance d
		for (IloNumVar d: this.dVarList) {
			IloNumExpr maxDev = this.cplex.constant(this.maxDeviation);
			this.cplex.addLe(d, maxDev);
		}
	}
	
	/**
	 * Method used to create the adversarial example by solving the model
	 * and print the status of the model after solving and the objective value
	 * @return	If the model was optimally solved or the optimality gap was reaches
	 * @throws IloException
	 */
	public boolean solve() throws IloException {
		// solve the model
		this.cplex.solve();
		
		// print the status after solving and the objective value
		System.out.println(this.cplex.getCplexStatus() + " " + this.cplex.getObjValue());
		
		// check if the model was optimally solved or the optimality gap was reached
		boolean solved = this.cplex.getCplexStatus().toString().contains("Optimal");
		return solved;
	}
	
	/**
	 * Method that returns the objective value(total disturbance)
	 * @return	The total disturbance
	 * @throws IloException
	 */
	public double getObj() throws IloException {
		return this.cplex.getObjValue();
	}
	
	/**
	 * Method that returns the gap between the best found feasible solution
	 * and the highest lower bound
	 * @return	The optimality gap
	 * @throws IloException
	 */
	public double getGap() throws IloException {
		return this.cplex.getMIPRelativeGap();
	}
	
	/**
	 * Method that returns the number of branching nodes used to solve the model
	 * @return	The number of branching nodes used
	 */
	public int getNodes() {
		return this.cplex.getIncumbentNode();
	}
	
	/**
	 * Method used to clean up and clear the model
	 * @throws IloException
	 */
	public void cleanup() throws IloException {
		this.cplex.clearModel();
		this.cplex.end();
	}
	
	/**
	 * Method that writes the created adversarial example as well as the original image 
	 * to a csv file as 28X28 arrays of doubles
	 * @param filename				Name of the file the adversarial example should be written to
	 * @param filenameOriginal		Name of the file the original image should be written to
	 * @throws UnknownObjectException
	 * @throws IloException
	 * @throws IOException
	 */
	public void createAdvExmpl(String filename, String filenameOriginal) throws UnknownObjectException, IloException, IOException{
		// open the files
		BufferedWriter w = new BufferedWriter(new FileWriter(filename));
		BufferedWriter wOriginal = new BufferedWriter(new FileWriter(filenameOriginal));
		
		// get the x variables corresponding to the input layer(the adversarial example)
		Layer inputLayer  = this.dnn.getLayers()[0];
		IloNumVar[] x_0 = this.xVarMap.get(inputLayer);
		
		// write the image as a 28X28 array of doubles
		for (int y=0; y < 28; y++) {
			for (int x=0; x < 28; x++) {
				w.write(this.cplex.getValue(x_0[28*y + x]) + "");
				wOriginal.write(this.input[28*y + x] + "");
				if (x < 27) {
					w.write(",");
					wOriginal.write(",");
				}
				else {
					w.write("\n");
					wOriginal.write("\n");
				}
			}
		}
		// close the writers
		w.close();
		wOriginal.close();
	}
	
	/**
	 * Method used to write the created adversarial example to a file as a 1d-array
	 * @param w		Writer that is used to write the adversarial example
	 * @throws UnknownObjectException
	 * @throws IloException
	 * @throws IOException
	 */
	public void writeAdvExmpl(BufferedWriter w) throws UnknownObjectException, IloException, IOException{
		// get the adversarial example
		Layer inputLayer  = this.dnn.getLayers()[0];
		IloNumVar[] x_0 = this.xVarMap.get(inputLayer);
		
		// write it to the file
		for (int y=0; y < x_0.length; y++) {
			w.write(Double.toString(this.cplex.getValue(x_0[y])));
			if (y < x_0.length - 1) {
				w.write(",");
			}
		}
		// go to the next line in the file so another example can be written
		w.write("\n");
	}
	
	/**
	 * Method used to get the activation of the neurons in the output layer
	 * @return
	 * @throws UnknownObjectException
	 * @throws IloException
	 */
	public double[] getOutput() throws UnknownObjectException, IloException{
		// create an array for the activations
		double[] output = new double[10];
		
		// get the x variables corresponding to the output layer
		Layer outputLayer  = this.dnn.getLayers()[this.dnn.getLayers().length-1];
		IloNumVar[] x_K = this.xVarMap.get(outputLayer);
		
		// store the value of the x variables in the array as a double
		for (int j=0; j < x_K.length; j++) {
			output[j] = this.cplex.getValue(x_K[j]);
		}
		
		return output;
	}
	
	/**
	 * Method used to print the output of the neurons in one of the layers
	 * @param k		The layer that should be printed
	 * @param x		Should the positive(x) values be printed or the negative(s)?(true prints x)
	 * @throws UnknownObjectException
	 * @throws IloException
	 */
	public void printOutput(int k, boolean x) throws UnknownObjectException, IloException{
		// get the x or s variables from the kth layer
		Layer layer_k  = this.dnn.getLayers()[k];
		double[] output = new double[layer_k.getN()];
		IloNumVar[] xs_k = null;
		if (x) {
			xs_k = this.xVarMap.get(layer_k);
		}
		else {
			xs_k = this.sVarMap.get(layer_k);
		}
		
		// store the values of the variables in a double array
		for (int j=0; j < xs_k.length; j++) {
			output[j] = this.cplex.getValue(xs_k[j]);
		}
		
		// print the array with the index of the layer
		System.out.println(k + ": " + Arrays.toString(output));
	}
	
}
