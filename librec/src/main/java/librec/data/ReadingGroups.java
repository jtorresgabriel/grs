package librec.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import happy.coding.io.FileIO;

public class ReadingGroups {

	private HashMap<Integer, List<String>> groupData;
	// path to data file
	private String dataPath;
	private HashMap<String, HashMap<Integer, String>> UserRatings;
	private HashMap<Integer, List<String>> ItemData;

	public ReadingGroups(String path) {
		dataPath = path;
	}
	public HashMap<Integer, List<String>> ReadingGroups(String path){
		
		// groupData has group number as Key and values list of user.
		try {
			BufferedReader br = FileIO.getReader(path);
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
				for (int i = 0; i < current.size(); i++) {
					if (user.equals(current.get(i)) == true) {
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
		return groupData;
	}
	
public HashMap<String, HashMap<Integer, String>>ReadUserRatings(String pathRatings, String pathPredictions){
		
		// reading individual ratings
		try {
			UserRatings = new HashMap<String, HashMap<Integer, String>>();
			ItemData = new HashMap<Integer, List<String>>();
			BufferedReader br = FileIO.getReader(pathRatings);
			
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

		BufferedReader br;
		try {
			br = FileIO.getReader(pathPredictions);
		
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
		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return UserRatings;

	}

public HashMap<Integer, List<String>>ReadItems(String pathRatings){
	try {
		
		ItemData = new HashMap<Integer, List<String>>();
		BufferedReader br = FileIO.getReader(pathRatings);
		
		String line = null;

		while ((line = br.readLine()) != null) {
			String[] data = line.split("[ \t,]");
			Integer itemId = Integer.parseInt(data[1]);
			String rate = data[2];
			
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
	return ItemData;

	
}


}
