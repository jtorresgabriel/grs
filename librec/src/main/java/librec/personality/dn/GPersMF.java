package librec.personality.dn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import happy.coding.io.FileIO;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;

//Copyright (C) 2014 Guibing Guo
//
//This file is part of LibRec.
//
//LibRec is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//LibRec is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

/**
 * Biased Matrix Factorization Models. <br/>
 * 
 * NOTE: To have more control on learning, you can add additional regularation
 * parameters to user/item biases. For simplicity, we do not do this.
 * 
 * @author guoguibing
 * 
 */
public class GPersMF extends IterativeRecommender {

	private int dimensions;
	private int values;

	// DenseMatrix to hold personality factor vectors for every dimension-value
	// combination
	// size of this densematrix would be ((dimensions*values) X no.of factors)
	// example: if personality data is of 5 dimensions and 7 possible values(1-7)
	// then densematrix size would be (35 X no.of factors)
	// every row represents the vector for every dimension-value combination
	// ex. row 0 represents vector for P11 (dimension 1, value 1),
	// row 1 represents vector for P12 (dimension 1, value 2) and so on.
	private DenseMatrix pers2dArray;

	// array to hold actual personality data of one user at a time for processing
	// purpose
	private double[] persDim;

	// Hashmap to store the personality data of all users in the input dataset;
	// key : userid, value: persDim
	private Map<Integer, double[]> persMap;

	public GPersMF(SparseMatrix rm, SparseMatrix tm, int fold) {
		super(rm, tm, fold);
	}

	/**
	 * initialize the model parameters and variables
	 */
	protected void initModel() throws Exception {

		super.initModel();

		// dimension input from config file
		dimensions = algoOptions.getInt("-d");

		// values input from config file
		values = algoOptions.getInt("-v");

		// determining the no.of rows of densematrix as dimensions * values
		int denseMatRows = dimensions * values;

		// declaring the size of densematrix
		pers2dArray = new DenseMatrix(denseMatRows, numFactors);

		// initializing all elements of dense matrix with value range between mean and
		// std
		pers2dArray.init(initMean, initStd);

		readData();

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);

		// initialize user bias
		userBias.init(initMean, initStd);

		// initialize item bias
		itemBias.init(initMean, initStd);

	}

	/**
	 * build the matrix factorization model
	 */
	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;

			for (MatrixEntry me : trainMatrix) {

				int u = me.row(); // user
				int j = me.column(); // item
				double ruj = me.get();

				double pred = predict(u, j);
				double euj = ruj - pred;

				loss += euj * euj;

				// update bias
				double bu = userBias.get(u);
				double sgd = euj - regB * bu;
				userBias.add(u, lRate * sgd);

				loss += regB * bu * bu;

				// update item bias
				double bi = itemBias.get(j);
				double sgdi = euj - regB * bi;
				itemBias.add(j, lRate * sgdi);

				loss += regB * bi * bi;

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					double newPersSum = getNewPersValues(u, f);

					double delta_u = euj * qjf - regU * puf;

					double delta_j = (euj * puf) + (euj * newPersSum) - (regI * qjf);

					P.add(u, f, lRate * delta_u);

					Q.add(j, f, lRate * delta_j);

					loss += regU * puf * puf + regI * qjf * qjf;

					updateNewPersVectors(u, euj, qjf, f);
				}

			}

			loss *= 0.5;

			if (isConverged(iter))
				break;

		} // end of training

	}

	/**
	 * predict the rating of item j by user u
	 */
	protected double predict(int u, int j) throws Exception {
		return globalMean + itemBias.get(j) + userBias.get(u) + dotProd(P, u, Q, j);
	}

	/**
	 * calculate the dot product of user and item factor matrices
	 * 
	 * @param m
	 * @param mrow
	 * @param n
	 * @param nrow
	 * @return
	 */
	public double dotProd(DenseMatrix m, int mrow, DenseMatrix n, int nrow) {
		int mcols = m.numColumns();
		int ncols = n.numColumns();
		assert mcols == ncols;

		double res = 0;
		for (int fac = 0; fac < mcols; fac++) {
			double persPlusUserFactor = 0.0;

			persPlusUserFactor = m.get(mrow, fac) + getNewPersValues(mrow, fac);
			res += persPlusUserFactor * n.get(nrow, fac);
		}
		return res;
	}

	/**
	 * Read the users personality data and store them in a hashmap with userid as
	 * the key and the personality dimensions acquired by FFM as the values
	 */
	private void readData() {
		try {

			BufferedReader br = FileIO.getReader(cf.getPath("dataset.personality"));
			persMap = new HashMap<Integer, double[]>();

			String line = null;
			while ((line = br.readLine()) != null) {
				String[] data = line.split(",");

				Integer userId = Integer.parseInt(data[0]);

				int p = 0;
				int persDimSize = data.length - 5;
				persDim = new double[persDimSize];

				for (int k = 5; k < data.length; k++) {
					persDim[p] = Double.parseDouble(data[k]);
					p++;
				}
				int userInnerId = rateDao.getUserId(userId.toString());
				persMap.put(userInnerId, persDim);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException iex) {
			iex.printStackTrace();
		}

	}

	/**
	 * get sum of personality factor vector values based on the respective actual
	 * personality data
	 * 
	 * @param a
	 * @param fac
	 * @return
	 */

	public double getNewPersValues(int a, int fac) {
		double persForUser[] = persMap.get(a);
		double persNewDimSum = 0;
		int denMatRowNum = 0;

		for (int d = 0; d < persForUser.length; d++) {

			double actualPersVal = persForUser[d];

			for (int x = 0; x < values; x++) {
				double valForComp = x + 1;

				if (Double.compare(actualPersVal, valForComp) == 0) {

					persNewDimSum += pers2dArray.get(denMatRowNum, fac);
				}
				denMatRowNum++;

			}

		}
		return persNewDimSum;
	}

	/**
	 * update personality factor vectors based on the respective actual personality
	 * data
	 * 
	 * @param u
	 * @param err
	 * @param qt
	 * @param fac
	 */
	public void updateNewPersVectors(int u, double err, double qt, int fac) {
		double persArray[] = persMap.get(u);
		int denMatRowNum = 0; // to be used to access the respective row in dense matrix

		for (int d = 0; d < persArray.length; d++) {

			double actualPersVal = persArray[d];

			for (int x = 0; x < values; x++) {
				double valForComp = x + 1;

				if (Double.compare(actualPersVal, valForComp) == 0) {
					double oldPersFacVal = pers2dArray.get(denMatRowNum, fac);
					pers2dArray.add(denMatRowNum, fac, lRate * (err * qt - regU * oldPersFacVal));
					loss += regU * oldPersFacVal * oldPersFacVal;
				}
				denMatRowNum++;
			}
		}
	}

}
