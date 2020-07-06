package webdata.compression;

import java.util.*;

public class ArithmeticCoding<KeyType extends Comparable<KeyType>> {

	public static float roundTo(float value, int places) {
		double scale = Math.pow(10, places);
		return (float) (Math.round(value * scale) / scale);
	}

	public float encode(List<KeyType> arr) {
		Restrict restricter = new Restrict(arr);
		for (KeyType value : arr) {
			restricter.updateRestrict(value);
		}
		return getMidValue(restricter);
	}

	private float getMidValue(Restrict restrict) {
		float mid = (restrict.high + restrict.low) / 2;
		for (int i = 0; i < 10; i++) {
			float roundedMid = roundTo(mid, i);
			if (restrict.low < roundedMid && roundedMid < restrict.high) {
				return roundedMid;
			}
		}
		return mid;
	}

	public List<KeyType> decode(List<KeyType> alphaBet, int size, float val) {
		Restrict restricter = new Restrict(alphaBet);
		List<KeyType> result = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			KeyType key = restricter.findKey(val);
			result.add(key);
		}
		return result;
	}

	private class Restrict {
		public float low = 0;
		public float high = 1;
		private Map<KeyType, Integer> history;
		private SortedSet<KeyType> keyTypeSortedSet;
		private int size;

		Restrict(List<KeyType> keys) {
			history = new HashMap<>();
			for (int i = 0; i < keys.size(); i++) {
				KeyType key = keys.get(i);
				history.putIfAbsent(key, 1);
			}
			keyTypeSortedSet = new TreeSet<>(history.keySet());
			size = keyTypeSortedSet.size();
		}

		public void updateRestrict(KeyType s_i) {
			float lowBound = getLowBound(s_i);
			float highBound = getHighBound(s_i, lowBound);
			float newLow = getNew(lowBound);
			float newHigh = getNew(highBound);
			low = newLow;
			high = newHigh;
			history.put(s_i, history.get(s_i) + 1);
			size += 1;
		}

		private float getNew(float bound) {
			float range = high - low;
			return low + range * bound;
		}

		private float getHighBound(KeyType s_i, float lowBound) {
			float pi = history.get(s_i) / (float) size;
			return lowBound + pi;
		}

		public float getLowBound(KeyType s_i) {
			int sum = 0;
			for (KeyType s : keyTypeSortedSet) {
				int value = history.get(s);
				if (s.compareTo(s_i) < 0) {
					sum += value;
				} else {
					break;
				}
			}
			return sum / (float) size;
		}

		public KeyType findKey(float val) {
			float lowBound = 0;
			for (KeyType alpha : this.keyTypeSortedSet) {
				float highBound = getHighBound(alpha, lowBound);
				float newLow = getNew(lowBound);
				float newHigh = getNew(highBound);
				if (newLow <= val && val < newHigh) {
					high = newHigh;
					low = newLow;
					history.put(alpha, history.get(alpha) + 1);
					size += 1;
					return alpha;
				}
				lowBound += history.get(alpha) / (float) size;
			}
			return null;
		}
	}

}
