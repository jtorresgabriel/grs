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
import java.util.Map.Entry;
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
public class Mvoted extends Recommender {


	protected float binThold;
	protected int[] columns;
	protected TimeUnit timeUnit;

	// Hashmap to store the all the data of all users in the input dataset (User,
	// Group);
	// key : userid, value: groupId

	private Map<Integer, List<String>> groupData;

	private HashMap<String, HashMap<Integer, String>> UserRatings;
	private HashMap<Integer, List<String>> ItemData;
	private ArrayList<Integer> missingGroup;

	public Mvoted(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		// TODO Auto-generated constructor stub
		try {
			readUserRatings();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		missingUser();
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
				String user = data[1].toLowerCase();

				List<String> current = groupData.get(groupId);
				if (current == null) {
					current = new ArrayList<String>();
					groupData.put(groupId, current);
				}
				int exite = 0;
				for (int i = 0; i <current.size(); i++) {
					if(user.equals(current.get(i)) == true) {
						 exite = 1;
					}
				}
				if (exite == 0) {
					current.add(user);	
				}
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
			ItemData = new HashMap<Integer, List<String>>();
			BufferedReader br = FileIO.getReader(cf.getPath("dataset.ratings"));
			
			String line = null;

			while ((line = br.readLine()) != null) {
				String[] data = line.split("[ \t,]");

				// HashMap<Integer,String>inner = new HashMap<Integer, String>();
				String key = data[0];
				Integer itemId = Integer.parseInt(data[1]);
				String rate = data[2];
				
				if (UserRatings.isEmpty() || !UserRatings.containsKey(key)) {
					HashMap<Integer, String> inner = new HashMap<Integer, String>();
					inner.put(itemId, rate);
					UserRatings.put(key, inner);
				} else if (UserRatings.containsKey(key)) {
					HashMap<Integer, String> inner = (HashMap<Integer, String>) UserRatings.get(key).clone();
					inner.put(itemId, rate);
					UserRatings.put(key, inner);
				}
				
				List<String> current = ItemData.get(itemId);
				if (current == null) {
					current = new ArrayList<String>();
					ItemData.put(itemId, current);
				}
				current.add(rate);
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

	protected void missingUser() {
		missingGroup = new ArrayList<Integer>();
		for (Entry<Integer, List<String>> entry : groupData.entrySet()) {
			// System.out.println(entry.getKey() + " = " + entry.getValue());
			int size = entry.getValue().size(); // user per group
			int missingUser = 0;
			for (int i = 0; i < size; i++) {
				if (UserRatings.containsKey(entry.getValue().get(i)) == false) {
					missingUser = missingUser + 1;
					//System.out.println(entry.getKey() + " , " + entry.getValue().get(i));
				}
			}
			if (missingUser == size) {
				missingGroup.add(entry.getKey());
				//System.out.print(missingGroup);
			
			}
		}
		for(Integer str : missingGroup) {
			groupData.remove(str);
		}
		
	}

	protected double averageMissing(int item) {
		int average = 0;

		for (int i = 0; i < ItemData.get(item).size(); i++) {
			average = average + Integer.parseInt(ItemData.get(item).get(i));
		}
		return average / ItemData.get(item).size();

	}
	protected double predict(int u, int j) {

		int group = Integer.parseInt(rateDao.getUserId(u));
		int item = Integer.parseInt(rateDao.getItemId(j));

		int size = 0;

		String users[] = null;
		int votes[] = null;
		double rate = 0;

		if (groupData.containsKey(group) == true) {
			size = groupData.get(group).size();
			users = new String[size];
			votes = new int[5];
				
			for (int i = 0; i < size; i++) {
				users[i] = groupData.get(group).get(i);
				if (UserRatings.get(users[i]) == null) {
					int y = ((int)Math.round(averageMissing(item))) -1;
					votes[y] = votes[y] + 1;
				} else {
					String x = (UserRatings.get(users[i]).get(item));
					int y = ((int)Math.round(Double.parseDouble(x))) - 1;
					votes[y] = votes[y] + 1;
				}
			}
			
				if ((size > 1) && (size % 2 != 0)) { //to check if the number is even or odd
					int a = votes[0];
					for (int z = 1; z < 5; z++) {
						int b = votes[z];
						if (b > a) {
							a = b;
							rate = z + 1;
						}
					}
				} else if ((size > 1) && (size % 2 == 0)) {
					int a = votes[0];
					for (int z = 1; z < 5; z++) {
						int b = votes[z];
						if (b > a) {
							a = b;
							rate = z + 1;
						} else if (b == a) {
							rate = z+1; //given the max rate if the vote are equal
						}
					}
				}
			}

		return rate;
	}
}