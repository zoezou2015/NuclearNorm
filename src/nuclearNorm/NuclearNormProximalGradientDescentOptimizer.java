/** Statistical Natural Language Processing System
Copyright (C) 2014-2015  Lu, Wei

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
* 
*/

package nuclearNorm;

import Jama.Matrix;
import Jama.SingularValueDecomposition;


/** @author yanyan
*
*/

public class NuclearNormProximalGradientDescentOptimizer {
	private double _learningRate = 1E-10;
	private Matrix _A;
	private Matrix _U;
	private Matrix _S;
	private Matrix _V;
	int _rank;
	private Matrix _g;
	private double _regularization;

	public NuclearNormProximalGradientDescentOptimizer(double learningRate, double regularization){
		this._learningRate = learningRate;
		this._regularization = regularization;
	}

	public double getLearningRate(){
		return this._learningRate;
	}

	public void setVariables(Matrix A){
		this._A = A;
	}

	public void setGradients(Matrix g){
		this._g = g;
	}
	
	public Matrix getVariables(){
		return this._A;
	}
	
	public Matrix getU(){
		return this._U;
	}
	
	public Matrix getS(){
		return this._S;
	}
	
	public Matrix getV(){
		return this._V;
	}

	public void SVD(){
		SingularValueDecomposition s = this._A.svd();
		StdOut.print("U = ");
		this._U = s.getU();  
		this._S = s.getS();
	    this._V = s.getV();
	    this._rank = s.rank();
	}
	
	public boolean optimize(){
		/*
		for(int i=0; i<this._w.length; i++){
			if (i is not emission){
				this._w[i] = this._w[i] - this._learningRate * this._g[i];
			}
		}
		*/
		for(int i=0; i<this._A.getRowDimension();i++){
			for(int j=0; j<this._A.getColumnDimension();j++){
				this._A.set(i, j, this._A.get(i, j)-this._learningRate*this._g.get(i,j));
				}
		}
		SVD();
		proxNuclear();
		
		return false;
	}

	public void proxNuclear() {
		double t = this._regularization*this._learningRate/2;
		for (int k = 0; k<this._rank; k++) {
			if (this._S.get(k,k) >= t) {
				this._S.set(k,k,this._S.get(k, k)-t+0.0);
			} else if (this._S.get(k,k) <= -t) {
				this._S.set(k, k, this._S.get(k, k)+t+0.0);
			} else {
				this._S.set(k, k, 0.0);
			}
		}
		this._A = this._U.times(this._S).times(this._V.transpose());
	}


}
