/**
 * 
 */
package librec.groups;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import happy.coding.io.FileIO;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.system.Dates;

import librec.data.DataDAO;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.Recommender;

/**
 * @author Jorge
 *
 */
public class Popularity extends Recommender {

	protected float binThold;
	protected int[] columns;
	protected TimeUnit timeUnit;

	// Hashmap to store the all the data of all users in the input dataset (User,
	// Group);
	// key : userid, value: groupId

	private Map<Integer, List<String>> groupData;

	private HashMap<String, HashMap<Integer, String>> UserRatings;
	
	public Popularity(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		// TODO Auto-generated constructor stub
		try {
			readUserRatings();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	protected void initModel() throws Exception {
		
		
	}
	
	// This funtion should be in java class to avoid repeticion
	public void readUserRatings() throws Exception {
			
		// groupData has group number as Key and values list of user.
		try {
			BufferedReader br = FileIO.getReader(cf.getPath("dataset.group"));
			groupData = new HashMap<Integer, List<String>>();

			String line = null;
			while ((line = br.readLine()) != null) {
				String[] data = line.split(",");

				Integer groupId = Integer.parseInt(data[0]);
				String user = data[1];

				List<String> current = groupData.get(groupId);
				if (current == null) {
					current = new ArrayList<String>();
					groupData.put(groupId, current);
				}
				current.add(user);
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException iex) {
			iex.printStackTrace();
		}

		// reading individual ratings
		try {
			UserRatings = new HashMap<String, HashMap<Integer, String>>();
			BufferedReader br = FileIO.getReader(cf.getPath("dataset.ratings"));
			;
			String line = null;

			while ((line = br.readLine()) != null) {
				String[] data = line.split("[ \t,]");

				// HashMap<Integer,String>inner = new HashMap<Integer, String>();
				String key = data[0];
				if (UserRatings.isEmpty() || !UserRatings.containsKey(key)) {
					HashMap<Integer, String> inner = new HashMap<Integer, String>();
					inner.put(Integer.parseInt(data[1]), data[2]);
					UserRatings.put(key, inner);
				} else if (UserRatings.containsKey(key)) {
					HashMap<Integer, String> inner =  (HashMap<Integer, String>) UserRatings.get(key).clone();
					inner.put(Integer.parseInt(data[1]), data[2]);
					UserRatings.put(key, inner);
				}
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException iex) {
			iex.printStackTrace();
		}

		// adding ratings from predict ratings to UserRatings
		
		BufferedReader br = FileIO.getReader(cf.getPath("dataset.ratings.predict"));
		
		String line = null;

		while ((line = br.readLine()) != null) {
			String[] data = line.split("[ \t,]");

			String key = data[0];
			if (UserRatings.isEmpty() || !UserRatings.containsKey(key)) {
				HashMap<Integer, String> inner = new HashMap<Integer, String>();
				inner.put(Integer.parseInt(data[1]), data[3]);
				UserRatings.put(key, inner);
			} else if (UserRatings.containsKey(key)) {
				HashMap<Integer, String> inner = (HashMap<Integer, String>) UserRatings.get(key).clone();
				inner.put(Integer.parseInt(data[1]), data[3]);
				UserRatings.put(key, inner);
			}
		}
		br.close();
		
	}

	protected double predict(int u, int j) {
		
		int group = Integer.parseInt(rateDao.getUserId(u));
		int item = Integer.parseInt(rateDao.getItemId(j));

		int size = 0;

		String users[] = null;
		double rate = 0;

		if (groupData.containsKey(group) == true) {
			size = groupData.get(group).size();
			users = new String[size];
			for (int i = 0; i < size; i++) {
				users[i] = groupData.get(group).get(i);
				System.out.print(users[i] + "," + item +",");
				String x = (UserRatings.get(users[i]).get(item));
				System.out.print( x + "\n");
				if (x == null) {
					System.out.print(users[i] + ";" + item + "\n");
					double y  = 3;
				} else {
					double y = Double.parseDouble(x);
				}
				double y = Double.parseDouble(x);
				if (y > rate) {
					rate = y;					
				}
			}
		}
		return rate;
	}
}