package reporting;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class JsonConverter {
	/**
	 * @author: Sheharyar Naseer (@sheharyarn)
	 * @license: MIT
	 */

	public static Object toJSON(Object object) {
		if (object instanceof HashMap) {
			JSONObject json = new JSONObject();
			HashMap map = (HashMap) object;
			for (Object key : map.keySet()) {
				json.put(key.toString(), toJSON(map.get(key)));
			}
			return json;
		} else if (object instanceof Iterable) {
			JSONArray json = new JSONArray();
			for (Object value : ((Iterable) object)) {
				json.add(toJSON(value));
			}
			return json;
		} else {
			return object;
		}
	}

	public static boolean isEmptyObject(JSONObject object) {
		return object.isEmpty();
	}

	public static HashMap<String, Object> getMap(JSONObject object, String key) {
		return toMap((JSONObject) object.get(key));
	}

	public static ArrayList getList(JSONObject object, String key) {
		return toList((JSONArray) object.get(key));
	}

	public static HashMap<String, Object> toMap(JSONObject object) {
		HashMap<String, Object> map = new HashMap();
		Set<String> keys = object.keySet();
		for (String key : keys) {
			map.put(key, fromJson(object.get(key)));
		}
		return map;
	}

	public static ArrayList toList(JSONArray array) {
		ArrayList list = new ArrayList();
		for (int i = 0; i < array.size(); i++) {
			list.add(fromJson(array.get(i)));
		}
		return list;
	}

	private static Object fromJson(Object json) {
		if (json == null) {
			return null;
		} else if (json instanceof JSONObject) {
			return toMap((JSONObject) json);
		} else if (json instanceof JSONArray) {
			return toList((JSONArray) json);
		} else {
			return json;
		}
	}
}