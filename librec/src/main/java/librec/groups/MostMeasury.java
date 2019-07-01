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
public class MostMeasury extends Recommender {

	protected float binThold;
	protected int[] columns;
	protected TimeUnit timeUnit;
	
	public static int a =0;

	// Hashmap to store the all the data of all users in the input dataset (User,
	// Group);
	// key : userid, value: groupId

	private Map<Integer, List<String>> groupData;

	private HashMap<String, HashMap<Integer, String>> UserRatings;
	private HashMap<Integer, List<String>> ItemData;
	private ArrayList<Integer> missingGroup;
	private ReadingGroups groupDataDao;

	public MostMeasury(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
				//System.out.print(missingGroup)
			}
		}
		for(Integer str : missingGroup) {
			groupData.remove(str);
		}
		
	}
	
	protected double averageMissing(int item) {
		double average = 0;

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
		double rates[] = null;
		int index = 0;

		if (groupData.containsKey(group) == true) {
			size = groupData.get(group).size();
			users = new String[size];
			rates = new double[size];
			
			for (int i = 0; i < size; i++) {
				users[i] = groupData.get(group).get(i);
				if (UserRatings.get(users[i]) == null) {
					rates[i] = averageMissing(item);
				}else {
				String x = (UserRatings.get(users[i]).get(item));
				rates[i] = Double.parseDouble(x);
				}
			}
		index = indexOfSmallest(rates);
		}
		return (rates[index]);
	}
	
	
	public static int indexOfSmallest(double[] array){
		
		 if ( array == null || array.length == 0 ) {
			 return -1; // null or empty
		 }

		  int smallest = 0;
		  for ( int i = 1; i < array.length; i++ )
		  {
		      if ( array[i] > array[smallest] ) smallest = i;
		  }
		  return smallest; // position of the first smallest found
		}
	}
