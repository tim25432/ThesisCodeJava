import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.UnknownObjectException;


/**
 * Class used to model the MILP formulation used to create a perturbation using training data
 * @author Tim Tjhay (495230tt)
 */
public class MILPPerturbation {
	private IloCplex cplex;

	private DNN dnn;
	private double[][] input;
	private int[] classification;

	private List<Map<Layer,IloNumVar[]>> xVarMaps;
	private List<Map<Layer,IloNumVar[]>> sVarMaps;
	private List<Map<Layer,IloNumVar[]>> zVarMaps;

	private IloNumVar[] pVarList;
	private IloNumVar[] qVarList;
	private IloNumVar[] yVarList;
	private IloNumVar[][] tVarList;

	private double[] yUB;
	
	private boolean addWeights;
	private boolean addDisturbance;


	/**
	 * Initializes the model that is used to create the perturbation
	 * @param dnn				The used DNN
	 * @param input				The training data images
	 * @param classification	The classification of the training data
	 * @param addWeights		If weights should be added
	 * @param addDisturbance	If disturbances should be added
	 * @param minDist			If the minimum distance variation should be applied
	 * @throws IloException
	 */
	public MILPPerturbation(DNN dnn, double[][] input, int[] classification, boolean addWeights, boolean addDisturbance, boolean minDist) throws IloException {
		this.cplex = new IloCplex();
//		this.cplex.setOut(null);

		this.dnn = dnn;
		this.input = input;
		this.classification = classification;

		this.xVarMaps = new LinkedList<>();
		this.sVarMaps = new LinkedList<>();
		this.zVarMaps = new LinkedList<>();

		this.pVarList = new IloNumVar[input[0].length];
		this.qVarList = new IloNumVar[input[0].length];
		this.yVarList = new IloNumVar[input.length];
		this.tVarList = new IloNumVar[input.length][10];

		this.yUB = new double[input.length];
		
		this.addWeights = addWeights;
		this.addDisturbance = addDisturbance;

		createVariables();
		addObjective(minDist);
		addYLB();
		addYUB();
		addTUB();

		addDefNeuron();
		addDefNeuron1();
		addXSZConstraints();

//		this.cplex.setParam(IloCplex.Param.MIP.Tolerances.MIPGap, 0.01);

//		this.cplex.setParam(IloCplex.Param.TimeLimit, 6000);

		this.cplex.setParam(IloCplex.Param.TimeLimit, 12 * 60 * 60);
	}

	/**
	 * Method used to create the variables
	 * @throws IloException
	 */
	private void createVariables() throws IloException {
		for (int h=0; h < this.input.length; h++) {
			this.xVarMaps.add(h, new HashMap<>());
			this.sVarMaps.add(h, new HashMap<>());
			this.zVarMaps.add(h, new HashMap<>());
			Map<Layer, IloNumVar[]> xVarMap = this.xVarMaps.get(h);
			Map<Layer, IloNumVar[]> sVarMap = this.sVarMaps.get(h);
			Map<Layer, IloNumVar[]> zVarMap = this.zVarMaps.get(h);
			for (Layer k: this.dnn.getLayers()) {
				if (k == this.dnn.getLayers()[0]) {
					continue;
				}
				xVarMap.put(k, new IloNumVar[k.getN()]);
				sVarMap.put(k, new IloNumVar[k.getN()]);
				zVarMap.put(k, new IloNumVar[k.getN()]);

				double[] xLB = k.getLowerBoundsX();
				double[] xUB = k.getUpperBoundsX();
				double[] sLB = k.getLowerBoundsS();
				double[] sUB = k.getUpperBoundsS();

				for (int i=0; i < k.getN(); i++) {
					xVarMap.get(k)[i] = this.cplex.numVar(xLB[i], xUB[i]);
					sVarMap.get(k)[i] = this.cplex.numVar(sLB[i], sUB[i]);
					if (k != this.dnn.getLayers()[0]) {
						zVarMap.get(k)[i] = this.cplex.boolVar();
					}
				}

				if (k == this.dnn.getLayers()[this.dnn.getNLayers()]) {
					this.yUB[h] = Arrays.stream(xUB).max().getAsDouble();
					this.yVarList[h] = this.cplex.numVar(0, this.yUB[h]);
				}
			}

			for (int i=0; i < this.tVarList[0].length; i++) {
				this.tVarList[h][i] = this.cplex.boolVar();
			}
		}

		for (int i=0; i < this.qVarList.length; i++) {
			this.pVarList[i] = this.cplex.numVar(0, Integer.MAX_VALUE);
			this.qVarList[i] = this.cplex.numVar(Integer.MIN_VALUE, Integer.MAX_VALUE);
		}
	}

	/**
	 * Method used to add the objective function
	 * @param minDist	If the minimum distance variation of the perturbation should be created
	 * @throws IloException
	 */
	private void addObjective(boolean minDist) throws IloException {
		IloNumExpr obj = this.cplex.constant(0);
		for (IloNumVar[] tList: this.tVarList) {
			IloNumExpr tSum = this.cplex.sum(tList);
			obj = this.cplex.sum(obj, tSum);
		}
		if (minDist) {
			IloNumExpr tot = this.cplex.constant(0);
			for (IloNumVar q: this.qVarList) {
				tot = this.cplex.sum(tot, this.cplex.abs(q));
			}
			for (IloNumVar p: this.pVarList) {
				tot = this.cplex.sum(tot, this.cplex.abs(this.cplex.diff(1, p)));
			}
			tot = this.cplex.prod(0.001, tot);
			obj = this.cplex.diff(obj, tot);
		}
		this.cplex.addMaximize(obj);
	}

	/**
	 * Method used to add the constraint that imposes the 
	 * definition of a neuron
	 * @throws IloException
	 */
	private void addDefNeuron() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		for (int h=0; h < this.input.length; h++) {
			for (int k=2; k < layers.length; k++) {
				double[][] w = layers[k].getWeights();
				double[] b = layers[k].getBias();

				IloNumVar[] x = this.xVarMaps.get(h).get(layers[k]);

				IloNumVar[] x_k_1 = this.xVarMaps.get(h).get(layers[k-1]);
				IloNumVar[] s = this.sVarMaps.get(h).get(layers[k]);
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
	}

	/**
	 * Method used to add the constraint that imposes the 
	 * definition of a neuron for the for the first hidden layer
	 * @throws IloException
	 */
	private void addDefNeuron1() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		for (int h=0; h < this.input.length; h++) {
			double[][] w = layers[1].getWeights();
			double[] b = layers[1].getBias();

			IloNumVar[] x_1 = this.xVarMaps.get(h).get(layers[1]);
			IloNumVar[] s = this.sVarMaps.get(h).get(layers[1]);

			IloNumVar[] p = this.pVarList;
			IloNumVar[] q = this.qVarList;
			for (int j=0; j < x_1.length; j++) {
				IloNumExpr rhs = this.cplex.diff(x_1[j], s[j]);
				IloNumExpr lhs = this.cplex.constant(b[j]);

				for (int i=0; i < p.length; i++) {
					IloNumExpr x_0  = this.cplex.constant(this.input[h][i]);
					if (this.addWeights) {
						x_0  = this.cplex.prod(x_0, p[i]);
					}
					if (this.addDisturbance) {
						x_0 = this.cplex.sum(x_0, q[i]);
					}
					IloNumExpr wx = this.cplex.prod(w[j][i], x_0);
					lhs = this.cplex.sum(lhs, wx);
				}

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
		for (int h=0; h < this.input.length; h++) {
			for (int k=1; k < layers.length; k++) {
				IloNumVar[] x = this.xVarMaps.get(h).get(layers[k]);
				IloNumVar[] s = this.sVarMaps.get(h).get(layers[k]);
				IloNumVar[] z = this.zVarMaps.get(h).get(layers[k]);
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
	}

	/**
	 * Method used to add constraint that imposes the lower bound on y
	 * @throws IloException
	 */
	private void addYLB() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		Layer outputLayer = layers[layers.length - 1];
		for (int h=0 ; h < this.input.length; h++) {
			for (int j=0; j < 10; j++) {
				IloNumExpr x = this.xVarMaps.get(h).get(outputLayer)[j];
				if (j != this.classification[h]) {
					x = this.cplex.prod(1.2, x);
				}
				this.cplex.addGe(this.yVarList[h], x);
			}
			this.cplex.addGe(this.yVarList[h], 0.01);
		}
	}

	/**
	 * Method used to add constraint that imposes the upper bound on y
	 * @throws IloException
	 */
	private void addYUB() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		Layer outputLayer = layers[layers.length - 1];
		for (int h=0 ; h < this.input.length; h++) {
			IloNumVar[] x_h_K = this.xVarMaps.get(h).get(outputLayer);
			for (int j=0; j < 10; j++) {
				IloConstraint indicator = this.cplex.eq(this.tVarList[h][j], 1);
				IloConstraint yxConstr = this.cplex.le(this.yVarList[h], x_h_K[j]);
				IloConstraint txyConstr = this.cplex.ifThen(indicator, yxConstr);
				this.cplex.add(txyConstr);
			}
		}
	}

	/**
	 * Method that is used to add the upper bound on the variable t
	 * @throws IloException
	 */
	private void addTUB() throws IloException {
		for (int h=0 ; h < this.input.length; h++) {
			for (int j=0; j < 10; j++) {
				IloNumExpr tau = this.cplex.constant(0);
				if (j == this.classification[h]) {
					tau = this.cplex.constant(1);
				}
				this.cplex.addLe(this.tVarList[h][j], tau);
			}
		}
	}

	/**
	 * Method used to create the perturbation by solving the model
	 * @return	If the model was solved to optimality
	 * @throws IloException
	 */
	public boolean solve() throws IloException {
		this.cplex.solve();
		boolean solved = this.cplex.getCplexStatus().toString().contains("Optimal");
		System.out.println(cplex.getCplexStatus() + " " + cplex.getObjValue());

		double tot = 0;
		if (this.addWeights) {
			for (IloNumVar p: this.pVarList) {
				tot += Math.abs(this.cplex.getValue(p));
			}
			System.out.println("p: " + tot);
		}
		
		tot = 0;
		if (this.addDisturbance) {
			for (IloNumVar q: this.qVarList) {
				tot += Math.abs(this.cplex.getValue(q));
			}
			System.out.println("q: " + tot);
		}
		
		tot = 0;
		for (IloNumVar[] tList: this.tVarList) {
			for (IloNumVar t: tList) {
				tot += this.cplex.getValue(t);
			}
		}
		System.out.println("t: " + tot);
		
		return solved;
	}

	/**
	 * Method used to write the perturbation to a csv file
	 * @param filename	Name of the file the perturbation should be written to
	 * @throws IOException
	 * @throws UnknownObjectException
	 * @throws IloException
	 */
	public void writePQ(String filename) throws IOException, UnknownObjectException, IloException {
		BufferedWriter w = new BufferedWriter(new FileWriter(filename));

		for (int i=0; i < this.pVarList.length; i++) {
			if (this.addWeights && this.addDisturbance) {
				double p = this.cplex.getValue(this.pVarList[i]);
				double q = this.cplex.getValue(this.qVarList[i]);
				w.write(p + "," + q + "\n");
			}
			else if (this.addWeights) {
				double p = this.cplex.getValue(this.pVarList[i]);
				w.write(p + "\n");
			}
			else if (this.addDisturbance) {
				double q = this.cplex.getValue(this.qVarList[i]);
				w.write(q + "\n");
			}
		}

		w.close();
	}

	/**
	 * Method used to print the output of the neurons in one of the layers
	 * @param k		The layer that should be printed
	 * @param x		Should the positive(x) values be printed or the negative(s)?(true prints x)
	 * @throws UnknownObjectException
	 * @throws IloException
	 */
	public void printOutput(int k, boolean x) throws UnknownObjectException, IloException{
		Layer layer_k  = this.dnn.getLayers()[k];
		double[] output = new double[layer_k.getN()];
		IloNumVar[] xs_k = null;
		for (int i=0; i < this.input.length; i ++) {
			if (x) {
				xs_k = this.xVarMaps.get(i).get(layer_k);
			}
			else {
				xs_k = this.sVarMaps.get(i).get(layer_k);
			}

			for (int j=0; j < xs_k.length; j++) {
				output[j] = this.cplex.getValue(xs_k[j]);
			}

			System.out.println(i + " " + k + ": " + Arrays.toString(output));
		}
	}

	/**
	 * Method used to print the values of t for every input in the training data
	 * @throws UnknownObjectException
	 * @throws IloException
	 */
	public void printOutputT() throws UnknownObjectException, IloException{
		for (int j=0; j<this.tVarList.length;j++) {
			IloNumVar[] t = this.tVarList[j];
			double[] l = new double[t.length];
			for (int i=0; i < t.length;i++) {
				l[i] = this.cplex.getValue(t[i]);
			}
			System.out.println(j + " t : " + Arrays.toString(l));
			System.out.println(j + " y : " + this.cplex.getValue(this.yVarList[j]) + ", class: " + this.classification[j]);
		}
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
