import java.util.HashMap;
import java.util.Map;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

/**
 * Class used to model the MILP formulation used to create adversarial examples
 * @author Tim Tjhay (495230tt)
 */
public class MILPAdvAfterImpr {
	private IloCplex cplex;
	
	private DNN dnn;
	private double[] input;
	private int targetDigit;
	private double[] p;
	private double[] q;
	private boolean perturb;
	
	private IloNumVar[] dVarList;
	private Map<Layer,IloNumVar[]> xVarMap;
	private Map<Layer,IloNumVar[]> sVarMap;
	private Map<Layer,IloNumVar[]> zVarMap;
	
	private IloConstraint[] targetConstr;
	private IloConstraint[] defDConstr;
	
	/**
	 * Initializes a model used to create an adversarial example
	 * after applying an accuracy improvement method
	 * @param dnn				The used DNN
	 * @param input				The original image
	 * @param targetDigit		The digit that the adversarial example needs to be classified as
	 * @param perturb			If the input should be perturbed
	 * @param perturbation		The perturbation that should be used
	 * @throws IloException		
	 */
	public MILPAdvAfterImpr(DNN dnn, double[] input, int targetDigit, boolean perturb, double[][] perturbation) throws IloException {
		this.cplex = new IloCplex();
		this.cplex.setOut(null);
		
		// store relevant data
		this.dnn = dnn;
		this.input = input;
		this.targetDigit = targetDigit;
		
		// store the perturbation if necessary
		this.perturb = perturb;
		if (this.perturb) {
			this.p = perturbation[0];
			this.q = perturbation[1];
		}
		
		// create maps for the variable
		this.dVarList = new IloNumVar[input.length];
		this.xVarMap = new HashMap<>();
		this.sVarMap = new HashMap<>();
		this.zVarMap = new HashMap<>();
		
		// create the variables and add the objective
		createVariables();
		addObjective();
		
		
		// add the constraints
		addDefNeuron1();
		addDefNeuron();
		
		addXSZConstraints();
		
		addTargetClass();
		addDefD();
		
		addMaxDeviation();
		
		// add a time limit of 300 seconds
		this.cplex.setParam(IloCplex.Param.TimeLimit, 300);
	}
	
	/**
	 * Method used to create the variables
	 * @throws IloException
	 */
	private void createVariables() throws IloException {
		for (Layer k: this.dnn.getLayers()) {
			this.xVarMap.put(k, new IloNumVar[k.getN()]);
			this.sVarMap.put(k, new IloNumVar[k.getN()]);
			if (k != this.dnn.getLayers()[0]) {
				this.zVarMap.put(k, new IloNumVar[k.getN()]);
			}
			double[] xLB = k.getLowerBoundsX();
			double[] xUB = k.getUpperBoundsX();
			double[] sLB = k.getLowerBoundsS();
			double[] sUB = k.getUpperBoundsS();
			
			for (int i=0; i < k.getN(); i++) {
				this.xVarMap.get(k)[i] = this.cplex.numVar(xLB[i], xUB[i]);
				this.sVarMap.get(k)[i] = this.cplex.numVar(sLB[i], sUB[i]);
				if (k != this.dnn.getLayers()[0]) {
					this.zVarMap.get(k)[i] = this.cplex.boolVar();
				}
			}
		}
		
		for (int i=0; i < this.dVarList.length; i++) {
			this.dVarList[i] = this.cplex.numVar(0, Integer.MAX_VALUE);
		}
	}
	
	/**
	 * Method used to add the objective function
	 * @throws IloException
	 */
	private void addObjective() throws IloException {
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
		for (int k=2; k < layers.length; k++) {
			double[][] w = layers[k].getWeights();
			double[] b = layers[k].getBias();

			IloNumVar[] x = this.xVarMap.get(layers[k]);

			IloNumVar[] x_k_1 = this.xVarMap.get(layers[k-1]);
			IloNumVar[] s = this.sVarMap.get(layers[k]);
			for (int j=0; j < x.length; j++) {
				IloNumExpr rhs = this.cplex.diff(x[j], s[j]);
				IloNumExpr lhs = this.cplex.constant(b[j]);

				for (int i=0; i < x_k_1.length; i++) {
					IloNumExpr wx = this.cplex.prod(w[j][i], x_k_1[i]);
					lhs = this.cplex.sum(lhs, wx);
				}

				this.cplex.addEq(lhs, rhs);

			}
		}
	}

	/**
	 * Method used to add the constraint that imposes the 
	 * definition of a neuron for the for the first hidden layer
	 * @throws IloException
	 */
	private void addDefNeuron1() throws IloException {
		// get the weights and biases of the first hidden layer
		Layer[] layers = this.dnn.getLayers();
		double[][] w = layers[1].getWeights();
		double[] b = layers[1].getBias();

		// get the needed x and s variables
		IloNumVar[] x_1 = this.xVarMap.get(layers[1]);
		IloNumVar[] s_1 = this.sVarMap.get(layers[1]);
		
		IloNumVar[] x_0 = this.xVarMap.get(layers[0]);

		// add the definition of the neurons
		for (int j=0; j < x_1.length; j++) {
			IloNumExpr rhs = this.cplex.diff(x_1[j], s_1[j]);
			IloNumExpr lhs = this.cplex.constant(b[j]);

			for (int i=0; i < x_0.length; i++) {
				IloNumExpr x_0_i  = x_0[i];
				if (this.perturb) {
					x_0_i  = this.cplex.prod(x_0_i, this.p[i]);
					x_0_i = this.cplex.sum(x_0_i, this.q[i]);
				}
				IloNumExpr wx = this.cplex.prod(w[j][i], x_0_i);
				lhs = this.cplex.sum(lhs, wx);
			}

			this.cplex.addEq(lhs, rhs);
		}
	}
	
	/**
	 * Method used to add the indicator constraints that use z
	 * @throws IloException
	 */
	private void addXSZConstraints() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		for (int k=1; k < layers.length; k++) {
			IloNumVar[] x = this.xVarMap.get(layers[k]);
			IloNumVar[] s = this.sVarMap.get(layers[k]);
			IloNumVar[] z = this.zVarMap.get(layers[k]);
			for (int j=0; j < x.length; j++) {
				IloConstraint zConstr1 = this.cplex.eq(z[j], 1);
				IloConstraint xConstr = this.cplex.eq(x[j], 0);
				IloConstraint xzConstr = this.cplex.ifThen(zConstr1, xConstr);
				this.cplex.add(xzConstr);
				
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
		Layer[] layers = this.dnn.getLayers(); 
		Layer outputLayer = layers[layers.length - 1];
		IloNumVar[] x_K = this.xVarMap.get(outputLayer);
		
		IloNumExpr lhs = x_K[this.targetDigit];

		this.targetConstr = new IloConstraint[x_K.length-1];
		int i = 0;
		
//		this.targetConstr = new IloConstraint[x_K.length];
//		this.targetConstr[0] = this.cplex.addGe(lhs, this.cplex.constant(0.01));
//		int i = 1;
		
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
		Layer inputLayer  = this.dnn.getLayers()[0];
		IloNumVar[] x_0 = this.xVarMap.get(inputLayer);
		
		this.defDConstr = new IloConstraint[this.dVarList.length*2];
		
		for (int j=0; j < this.dVarList.length; j++) {
			IloNumExpr lhs = this.dVarList[j];
			IloNumExpr rhs = this.cplex.diff(x_0[j], this.input[j]);
			
			this.defDConstr[2*j] = this.cplex.addGe(lhs, rhs);
			lhs = this.cplex.prod(this.cplex.constant(-1), lhs);
			this.defDConstr[2*j + 1] =  this.cplex.addLe(lhs, rhs);
		}
	}
	
	/**
	 * Method used to add the constraint that imposes the maximum deviation
	 * @throws IloException
	 */
	private void addMaxDeviation() throws IloException {
		for (IloNumVar d: this.dVarList) {
			IloNumExpr maxDev = this.cplex.constant(1);
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
		this.cplex.solve();
		System.out.println(this.cplex.getCplexStatus() + " " + this.cplex.getObjValue());
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
	
}
