import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.UnknownObjectException;


/**
 * Class used to model the MILP formulation used to create feature visualization
 * @author Tim Tjhay (495230tt)
 */
public class MILPVisualize {
	private IloCplex cplex;
	
	private DNN dnn;

	private Map<Layer,IloNumVar[]> xVarMap;
	private Map<Layer,IloNumVar[]> sVarMap;
	private Map<Layer,IloNumVar[]> zVarMap;
	
	private int targetDigit;
	
	/**
	 * Initializes the model used to create a visualization of the target digit
	 * @param dnn			The used DNN
	 * @param targetDigit	The digit the feature visualization is made of
	 * @throws IloException
	 */
	public MILPVisualize(DNN dnn, int targetDigit) throws IloException {
		this.cplex = new IloCplex();
		
		this.dnn = dnn;

		this.xVarMap = new HashMap<>();
		this.sVarMap = new HashMap<>();
		this.zVarMap = new HashMap<>();
		
		this.targetDigit = targetDigit;
		
		createVariables();
		
		addDefNeuron();
		addXSZConstraints();
		
		addObjective();
		
		this.cplex.setOut(null);
	}

	/**
	 * Method used to create the variables
	 * @throws IloException
	 */
	private void createVariables() throws IloException {
		for (Layer k: this.dnn.getLayers()) {
			xVarMap.put(k, new IloNumVar[k.getN()]);
			sVarMap.put(k, new IloNumVar[k.getN()]);
			if (k != this.dnn.getLayers()[0]) {
				zVarMap.put(k, new IloNumVar[k.getN()]);
			}
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
		}
	}
	
	/**
	 * Method used to add the objective function
	 * @throws IloException
	 */
	private void addObjective() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		Layer outputLayer = layers[layers.length-1];
		IloNumVar x = this.xVarMap.get(outputLayer)[this.targetDigit]; 
		this.cplex.addMaximize(x);
	}
	
	/**
	 * Method used to add the constraint that imposes the 
	 * definition of a neuron
	 * @throws IloException
	 */
	private void addDefNeuron() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		for (int k=1; k < layers.length; k++) {
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
				
				cplex.addEq(lhs, rhs);
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
	 * Method used to write the feature visualization to a file
	 * @param filename		Name of the file the visualization should be written to
	 * @throws UnknownObjectException
	 * @throws IloException
	 * @throws IOException
	 */
	public void createVisualization(String filename) throws UnknownObjectException, IloException, IOException{
		BufferedWriter w = new BufferedWriter(new FileWriter(filename));
		
		Layer inputLayer  = this.dnn.getLayers()[0];
		IloNumVar[] x_0 = this.xVarMap.get(inputLayer);
		
		for (int y=0; y < 28; y++) {
			for (int x=0; x < 28; x++) {
				w.write(this.cplex.getValue(x_0[28*y + x]) + "");
				if (x < 27) {
					w.write(",");
				}
				else {
					w.write("\n");
				}
			}
		}
		w.close();
	}
	
	/**
	 * Method used to create the visualization by solving the model
	 * @throws IloException
	 */
	public void solve() throws IloException {
		this.cplex.solve();
		
		Layer[] layers = this.dnn.getLayers();
		Layer outputLayer = layers[layers.length-1];
		IloNumVar[] x_K = this.xVarMap.get(outputLayer);
		
		for (IloNumVar x: x_K) {
			System.out.print(cplex.getValue(x) + " ");
		}
		System.out.println();
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
