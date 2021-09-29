import java.util.HashMap;
import java.util.Map;

import ilog.concert.IloConstraint;
import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.concert.IloObjective;
import ilog.cplex.IloCplex;


/**
 * Class used to model the MILP formulation used to calculate tightened bounds
 * @author Tim Tjhay (495230tt)
 */
public class MILPBounds {
	private IloCplex cplex;
	
	private DNN dnn;

	private Map<Layer,IloNumVar[]> xVarMap;
	private Map<Layer,IloNumVar[]> sVarMap;
	private Map<Layer,IloNumVar[]> zVarMap;
	
	private IloObjective obj;
	
	public MILPBounds(DNN dnn, boolean timeLimit) throws IloException {
		this.cplex = new IloCplex();
		
		this.dnn = dnn;

		this.xVarMap = new HashMap<>();
		this.sVarMap = new HashMap<>();
		this.zVarMap = new HashMap<>();
		
		createVariables();
		
		addDefNeuron();
		addXSZIndicator();
		
		if (timeLimit) {
			this.cplex.setParam(IloCplex.Param.TimeLimit, 1);
		}
		
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
	 * Method used to add maximizing the x variable of 
	 * the neuron in the last layer as the objective
	 * @throws IloException
	 */
	private void addObjectiveX() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		Layer toMax = layers[layers.length-1];
		IloNumVar x = this.xVarMap.get(toMax)[0]; 
		this.obj = this.cplex.addMaximize(x);
	}
	
	/**
	 * Method used to add maximizing the x variable of 
	 * the neuron in the last layer as the objective
	 * @throws IloException
	 */
	private void addObjectiveS() throws IloException {
		Layer[] layers = this.dnn.getLayers();
		Layer toMax = layers[layers.length-1];
		IloNumVar s = this.sVarMap.get(toMax)[0]; 
		this.obj = this.cplex.addMaximize(s);
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
	private void addXSZIndicator() throws IloException {
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
	 * Method used to get the upper bound on the 
	 * x and s variable of the neuron in the last layer
	 * @return	The upper bounds
	 * @throws IloException
	 */
	public double[] getUpperBounds() throws IloException {
		double[] boundsXS = new double[2];
		// get the upper bound on x
		addObjectiveX();
		this.cplex.solve();
		boundsXS[0] = this.cplex.getBestObjValue();
		this.cplex.delete(this.obj);
		this.cplex.clearCuts();
		// get the upper bound on s
		addObjectiveS();
		this.cplex.solve();
		boundsXS[1] = this.cplex.getBestObjValue();
		return boundsXS;
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
