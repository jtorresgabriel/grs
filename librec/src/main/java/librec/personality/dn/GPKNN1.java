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

package librec.personality.dn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Strings;
import happy.coding.math.Stats;
import librec.data.Configuration;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.intf.Recommender;

/**
 * <h3>User-based Nearest Neighbors</h3>
 * 
 * <p>
 * It supports both recommendation tasks: (1) rating prediction; and (2) item
 * ranking (by configuring {@code item.ranking=on} in the librec.conf). For item
 * ranking, the returned score is the summation of the similarities of nearest
 * neighbors.
 * </p>
 * 
 * <p>
 * When the number of users is extremely large which makes it memory intensive
 * to store/precompute all user-user correlations, a trick presented by (Jahrer
 * and Toscher, Collaborative Filtering Ensemble, JMLR 2012) can be applied.
 * Specifically, we can use a basic SVD model to obtain user-feature vectors,
 * and then user-user correlations can be computed by Eqs (17, 15).
 * </p>
 * 
 * @author guoguibing
 * 
 */
@Configuration("knn, similarity, shrinkage")
public class GPKNN1 extends Recommender {

	// user: nearest neighborhood
	private double alpha;
	private SymmMatrix userCorrs;
	private SymmMatrix persCorrs;
	private SymmMatrix combCorrs;
	private DenseVector userMeans;

	private Map<Integer, double[]> persMap;

	// array to hold actual personality data of one user at a time for processing
	// purpose
	private double[] persDim;

	public GPKNN1(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	@Override
	protected void initModel() throws Exception {

		alpha = algoOptions.getFloat("-alpha");

		readData();

		userCorrs = buildCorrs(true);

		buildPersCorrs();

		combineSimilarities();

		userMeans = new DenseVector(numUsers);

		for (int u = 0; u < numUsers; u++) {
			SparseVector uv = trainMatrix.row(u);
			userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
		}
	}

	@Override
	protected double predict(int u, int j) {

		// find a number of similar users
		Map<Integer, Double> nns = new HashMap<>();
		SparseVector dv = combCorrs.row(u); // use combined similarity

		for (int v : dv.getIndex()) {
			double sim = dv.get(v);
			double rate = trainMatrix.get(v, j);

			if (isRankingPred && rate > 0)
				nns.put(v, sim); // similarity could be negative for item ranking
			else if (sim > 0 && rate > 0)
				nns.put(v, sim);
		}

		// topN similar users
		if (knn > 0 && knn < nns.size()) {
			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
			nns.clear();
			for (Map.Entry<Integer, Double> kv : subset)
				nns.put(kv.getKey(), kv.getValue());
		}

		if (nns.size() == 0)
			return isRankingPred ? 0 : globalMean;

		if (isRankingPred) {
			// for item ranking

			return Stats.sum(nns.values());
		} else {
			// for rating prediction

			double sum = 0, ws = 0;
			for (Entry<Integer, Double> en : nns.entrySet()) {
				int v = en.getKey();
				double sim = en.getValue();
				double rate = trainMatrix.get(v, j);

				sum += sim * (rate - userMeans.get(v));
				ws += Math.abs(sim);
			}

			return ws > 0 ? userMeans.get(u) + sum / ws : globalMean;
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
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
	 * Build personality based similarity matrix
	 * 
	 */
	public void buildPersCorrs() {

		persCorrs = new SymmMatrix(persMap.size());

		for (int row = 0; row < persMap.size(); row++) {
			SparseVector iv = new SparseVector(persMap.get(row).length, persMap.get(row));

			for (int col = row + 1; col < persMap.size(); col++) {
				SparseVector jv = new SparseVector(persMap.get(col).length, persMap.get(col));

				double sim = correlation(iv, jv);

				if (!Double.isNaN(sim)) {

					persCorrs.set(row, col, sim);

				}
			}
		}

	}

	/**
	 * Combine the personality based and rating based similarities by assigning
	 * weightage to each
	 */
	public void combineSimilarities() {

		combCorrs = new SymmMatrix(numUsers);

		for (int a = 0; a < numUsers; a++) {
			for (int b = a + 1; b < numUsers; b++) {
				double combinedSimilarity = (userCorrs.get(a, b) * alpha) + (persCorrs.get(a, b) * (1.0 - alpha));
				combCorrs.set(a, b, combinedSimilarity);

			}

		}

	}
}
