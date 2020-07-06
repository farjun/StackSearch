package webdata.utils;


import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class SparseVector {

	Map<String, Double> data;

	// ======================================== ctor =============================================


	public SparseVector() {
		this.data = new HashMap<>();
	}

	SparseVector(int dataSize) {
		this.data = new HashMap<>(dataSize);
	}

	SparseVector(Map<String, Double> data) {
		this(data.size());
		this.data.putAll(data);
	}

	public SparseVector(SparseVector vector) {
		this(vector.data);
	}

	@Override
	public String toString() {
		return this.data.toString();
	}

	// ======================================== inner data =============================================

	public void addValue(String key, Double value) {
		data.put(key, value);
	}

	public boolean containsKey(String key) {
		return data.containsKey(key);
	}

	// ======================================== values =============================================

	public double prod() {
		Set<String> keySet = this.data.keySet();
		if (keySet.size() == 0) {
			return 0;
		}
		double s = 1;
		for (String key : keySet) {
			s *= this.data.get(key);
		}
		return s;
	}

	public double prod(int size) {
		Set<String> keySet = this.data.keySet();
		if (keySet.size() == 0 || keySet.size() != size) { // size 0 or got 0 value implicit.
			return 0;
		}
		double s = 1;
		for (String key : keySet) {
			s *= this.data.get(key);
		}
		return s;
	}

	public double dot(SparseVector vector) {
		if (this.data.size() > vector.data.size()) {
			return vector.dot(this);
		}
		double s = 0;
		Set<String> keySet = this.data.keySet();
		for (String key : keySet) {
			if (vector.data.containsKey(key)) {
				s += this.data.get(key) * vector.data.get(key);
			}
		}
		return s;
	}

	public double magnitude() {
		double scale = 0;
		Set<Map.Entry<String, Double>> entries = this.data.entrySet();
		for (Map.Entry<String, Double> entry : entries) {
			Double value = entry.getValue();
			scale += value * value;
		}
		scale = Math.sqrt(scale);
		return scale;
	}

	// ======================================== operations =============================================

	public SparseVector inner(SparseVector vector) {
		if (this.data.size() > vector.data.size()) {
			return vector.inner(this);
		}
		SparseVector newVec = new SparseVector(this.data);
		newVec.innerUpdate(vector);
		return newVec;
	}

	public void innerUpdate(SparseVector vector) {
		if (this.data.size() > vector.data.size()) {
			Set<String> keySet = vector.data.keySet();
			for (String key : keySet) {
				if (this.containsKey(key)) {
					this.addValue(key, this.data.get(key) * vector.data.get(key));
				}
			}
		} else {
			Set<String> keySet = this.data.keySet();
			for (String key : keySet) {
				if (vector.containsKey(key)) {
					this.addValue(key, this.data.get(key) * vector.data.get(key));
				} else {
					this.data.remove(key);
				}
			}
		}
	}

	public SparseVector plus(SparseVector vector) {
		SparseVector newVec = new SparseVector(this.data);
		newVec.plusUpdate(vector);
		return newVec;
	}

	public void plusUpdate(SparseVector vector) {
		Set<Map.Entry<String, Double>> entries = vector.data.entrySet();
		for (Map.Entry<String, Double> entry : entries) {
			this.data.compute(entry.getKey(), (key, value) -> {
				if (value == null) {
					return vector.data.get(key);
				} else {
					return vector.data.get(key) + value;
				}
			});
		}
	}

	// ======================================== Apply =============================================

	public void applyMulByScale(double scale) {
		if (scale == 1) {
			return; // Scale == 1 so we don't need to do anything.
		}
		if (scale == 0) { // Zero out the vector.
			this.data.clear();
			return;
		}
		Set<Map.Entry<String, Double>> entries = this.data.entrySet();
		for (Map.Entry<String, Double> entry : entries) {
			this.addValue(entry.getKey(), entry.getValue() * scale);
		}
	}

	public void applyDivByScale(double scale) {
		if (scale == 1) {
			return; // Scale == 1 so we don't need to do anything.
		}
		Set<Map.Entry<String, Double>> entries = this.data.entrySet();
		for (Map.Entry<String, Double> entry : entries) {
			this.addValue(entry.getKey(), entry.getValue() / scale);
		}
	}

	public void applyPower(double scale) {
		if (scale == 1) {
			return; // Scale == 1 so we don't need to do anything.
		}
		Set<Map.Entry<String, Double>> entries = this.data.entrySet();
		for (Map.Entry<String, Double> entry : entries) {
			this.addValue(entry.getKey(), Math.pow(entry.getValue(), scale));
		}
	}


}

