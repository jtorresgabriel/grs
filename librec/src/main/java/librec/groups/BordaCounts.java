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
import librec.data.ReadingGroups;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.Recommender;

/**
 * @author Jorge
 *
 */
public class BordaCounts extends Recommender {


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
	private ReadingGroups groupDataDao;

	public BordaCounts(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		groupDataDao = new ReadingGroups(cf.getPath("dataset.ratings.group"));
		groupData = groupDataDao.ReadingGroups(cf.getPath("dataset.group"));
		UserRatings = groupDataDao.ReadUserRatings(cf.getPath("dataset.ratings"), cf.getPath("dataset.ratings.predict"));
		ItemData = groupDataDao.ReadItems(cf.getPath("dataset.ratings"));
		missingUser();
	}

	@Override
	protected void initModel() throws Exception {

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
		int borda = 0;

		if (groupData.containsKey(group) == true) {
			size = groupData.get(group).size();
			users = new String[size];
			votes = new int[5];
				
			for (int i = 0; i < size; i++) {
				users[i] = groupData.get(group).get(i);
				if (UserRatings.get(users[i]) == null) {
					int y = ((int)Math.round(averageMissing(item))) -1;
					votes[y] = votes[y] + 1;
				}else if ((UserRatings.get(users[i]).get(item)) == null) {
						int y = ((int)Math.round(averageMissing(item))) -1;
						votes[y] = votes[y] + 1;
				} else if  ((UserRatings.get(users[i]).get(item)).equals("0")) {
					int y = ((int)Math.round(averageMissing(item))) -1;
					votes[y] = votes[y] + 1;
				} else {
					String x = (UserRatings.get(users[i]).get(item));
					int y = ((int)Math.round(Double.parseDouble(x))) - 1;
					votes[y] = votes[y] + 1;
					}
				}
			for (int i = 0; i < size; i++) {
				borda = borda + (i+1) * votes[i];
			}
			return borda;
			}

		return borda;
	}
}