package webdata;

import webdata.utils.SparseVector;

import java.util.*;

public class ReviewSearch {
	private final IndexReader iReader;

	/**
	 * Constructor
	 */
	public ReviewSearch(IndexReader iReader) {
		this.iReader = iReader;
	}

	public static void main(String[] args) {
		SparseVectorTF qVectorTF = new SparseVectorTF("l");
		SparseVectorIDF qVectorIDF = new SparseVectorIDF("t", 1000000);
		qVectorTF.addValue("auto", 0d);
		qVectorTF.addValue("best", 1d);
		qVectorTF.addValue("car", 1d);
		qVectorTF.addValue("insurance", 1d);

		qVectorIDF.addValue("auto", 5000d);
		qVectorIDF.addValue("best", 50000d);
		qVectorIDF.addValue("car", 10000d);
		qVectorIDF.addValue("insurance", 1000d);

		SparseVectorNormalized qVectorNormalized = new SparseVectorNormalized(qVectorTF, qVectorIDF, "c");

		SparseVectorTF dVectorTF = new SparseVectorTF("l");
		SparseVectorIDF dVectorIDF = new SparseVectorIDF("n", 1000000);
		dVectorTF.addValue("auto", 1d);
		dVectorTF.addValue("best", 0d);
		dVectorTF.addValue("car", 1d);
		dVectorTF.addValue("insurance", 2d);
		dVectorIDF.addValue("auto", 5000d);
		dVectorIDF.addValue("best", 50000d);
		dVectorIDF.addValue("car", 10000d);
		dVectorIDF.addValue("insurance", 1000d);
		SparseVectorNormalized dVectorNormalized = new SparseVectorNormalized(dVectorTF, dVectorIDF, "c");

		double score = dVectorNormalized.dot(qVectorNormalized);
		System.out.println(score);
	}

	/**
	 * Returns a list of the id-s of the k most highly ranked reviews for the
	 * given query, using the vector space ranking function lnn.ltc (using the
	 * SMART notation)
	 * The list should be sorted by the ranking
	 */
	public Enumeration<Integer> vectorSpaceSearch(Enumeration<String> query, int k) {
		return vectorSpaceSearchWithParams(query, k,
				"l", "n", "n",
				"l", "t", "c"
		);
	}

	public Enumeration<Integer> betterVectorSpaceSearch(Enumeration<String> query, int k) {
		return vectorSpaceSearchWithParams(query, k,
				"l", "s", "c",
				"l", "s", "c"
		);
	}

	private Enumeration<Integer> vectorSpaceSearchWithParams(Enumeration<String> query, int k, String dTF, String dIDF, String dNormalize, String qTF, String qIDF, String qNormalize) {
		int numberOfReviews = this.iReader.getNumberOfReviews();
		Map<String, Integer> tfRaw = getQueryTF(query);
		SparseVectorTF queryTF = new SparseVectorTF(qTF);
		SparseVectorIDF queryIDF = new SparseVectorIDF(qIDF, numberOfReviews);
		SparseVectorIDF documentsIDF = new SparseVectorIDF(dIDF, numberOfReviews);
		Map<Integer, SparseVectorTF> documentsTF = new HashMap<>();
		for (String token : tfRaw.keySet()) {
			Enumeration<Integer> reviewsWithToken = this.iReader.getReviewsWithToken(token);
			// doc TF
			double tokenFreq = 0;
			while (reviewsWithToken.hasMoreElements()) {
				int rid = reviewsWithToken.nextElement();
				Integer freqInRid = reviewsWithToken.nextElement();
				SparseVectorTF ridTF = documentsTF.computeIfAbsent(rid, (id) -> new SparseVectorTF(dTF));
				ridTF.addValue(token, freqInRid.doubleValue());
				tokenFreq++;
			}
			// this part is for the IDF
			queryIDF.addValue(token, tokenFreq);
			documentsIDF.addValue(token, tokenFreq);
		}
		for (Map.Entry<String, Integer> entry : tfRaw.entrySet()) {
			queryTF.addValue(entry.getKey(), entry.getValue().doubleValue());
		}
		SparseVectorNormalized queryNormalized = new SparseVectorNormalized(queryTF, queryIDF, qNormalize);
		Map<Integer, SparseVectorNormalized> documentsNormalized = new HashMap<>();
		for (Map.Entry<Integer, SparseVectorTF> entry : documentsTF.entrySet()) {
			documentsNormalized.put(entry.getKey(),
					new SparseVectorNormalized(entry.getValue(), documentsIDF, dNormalize));
		}
		Comparator<PriorityItemInt> integerPriorityItemComparator = new PriorityItemComparator();
		PriorityQueue<PriorityItemInt> pq = new PriorityQueue<>(integerPriorityItemComparator);
		Iterator<Map.Entry<Integer, SparseVectorNormalized>> iterator = documentsNormalized.entrySet().iterator();
		for (int i = 0; i < k && iterator.hasNext(); i++) {
			Map.Entry<Integer, SparseVectorNormalized> entry = iterator.next();
			pq.add(new PriorityItemInt(entry.getKey(), entry.getValue().dot(queryNormalized)));

		}
		while (iterator.hasNext()) {
			Map.Entry<Integer, SparseVectorNormalized> entry = iterator.next();
			PriorityItemInt lowPriorityItem = pq.peek();
			PriorityItemInt priorityItemInt = new PriorityItemInt(entry.getKey(), entry.getValue().dot(queryNormalized));
			if (integerPriorityItemComparator.compare(lowPriorityItem, priorityItemInt) < 0) {
				pq.poll();
				pq.add(priorityItemInt);
			}
		}
		PriorityQueue<PriorityItemInt> reversedPq = new PriorityQueue<>(integerPriorityItemComparator.reversed());
		reversedPq.addAll(pq);
		return new PQEnumeration(reversedPq);
	}

	private Map<String, Integer> getQueryTF(Enumeration<String> query) {
		Map<String, Integer> tfRaw = new HashMap<>();
		while (query.hasMoreElements()) {
			String token = query.nextElement();
			tfRaw.compute(token, (dummy, value) -> value != null ? value + 1 : 1);
		}
		return tfRaw;
	}

	/**
	 * Returns a list of the id-s of the k most highly ranked reviews for the
	 * given query, using the language model ranking function, smoothed using a
	 * mixture model with the given value of lambda
	 * The list should be sorted by the ranking
	 */
	public Enumeration<Integer> languageModelSearch(Enumeration<String> query, double lambda, int k) {
		SparseVector PTMC = new SparseVector();
		Map<Integer, SparseVector> PTMDs = new HashMap<>();
		Map<String, Integer> tfRaw = this.getQueryTF(query);
		for (String token : tfRaw.keySet()) {
			Enumeration<Integer> reviewsWithToken = iReader.getReviewsWithToken(token);
			double mc = 0;
			while (reviewsWithToken.hasMoreElements()) {
				int rid = reviewsWithToken.nextElement();
				double freqInRid = reviewsWithToken.nextElement();
				SparseVector PTMDRid = PTMDs.computeIfAbsent(rid, (dummy) -> new SparseVector());
				PTMDRid.addValue(token, freqInRid);
				mc += freqInRid;
			}
			PTMC.addValue(token, mc);
		}
		PTMC.applyDivByScale(this.iReader.getTokenSizeOfReviews());
		PTMC.applyMulByScale(1 - lambda);
		for (Map.Entry<Integer, SparseVector> entry : PTMDs.entrySet()) {
			SparseVector sparseVector = entry.getValue();
			sparseVector.applyDivByScale(this.iReader.getReviewLength(entry.getKey()));
			sparseVector.applyMulByScale(lambda);
			sparseVector.plusUpdate(PTMC);
		}
		Map<Integer, SparseVector> probabilityVectors = PTMDs;
		Comparator<PriorityItemInt> integerPriorityItemComparator = new PriorityItemComparator();
		PriorityQueue<PriorityItemInt> pq = new PriorityQueue<>(integerPriorityItemComparator);
		Iterator<Map.Entry<Integer, SparseVector>> iterator = probabilityVectors.entrySet().iterator();
		for (int i = 0; i < k && iterator.hasNext(); i++) {
			Map.Entry<Integer, SparseVector> entry = iterator.next();
			pq.add(new PriorityItemInt(entry.getKey(), entry.getValue().prod(tfRaw.size())));
		}
		while (iterator.hasNext()) {
			Map.Entry<Integer, SparseVector> entry = iterator.next();
			PriorityItemInt lowPriorityItem = pq.peek();
			PriorityItemInt priorityItemInt = new PriorityItemInt(entry.getKey(), entry.getValue().prod(tfRaw.size()));
			if (integerPriorityItemComparator.compare(lowPriorityItem, priorityItemInt) < 0) {
				pq.poll();
				pq.add(priorityItemInt);
			}
		}
		PriorityQueue<PriorityItemInt> reversedPq = new PriorityQueue<>(integerPriorityItemComparator.reversed());
		reversedPq.addAll(pq);
		return new PQEnumeration(reversedPq);
	}

	/**
	 * Returns a list of the id-s of the k most highly ranked productIds for the
	 * given query using a function of your choice
	 * The list should be sorted by the ranking
	 */
	public Collection<String> productSearch(Enumeration<String> query, int k) {
		Set<Integer> rids = new HashSet<>();
		Map<String, Set<Integer>> productIdToReviewsIds = new HashMap<>();
		updateProductIdToReviewsIds(this.betterVectorSpaceSearch(query, 6 * k), rids, productIdToReviewsIds);
		updateProductIdToReviewsIds(this.languageModelSearch(query, 0.2, 2 * k), rids, productIdToReviewsIds);
		updateProductIdToReviewsIds(this.languageModelSearch(query, 0.5, 2 * k), rids, productIdToReviewsIds);
		updateProductIdToReviewsIds(this.languageModelSearch(query, 0.8, 2 * k), rids, productIdToReviewsIds);
		int helpfulnessAddend = 3;
		int scoreAddend = 3;
		PriorityItemStringComparator priorityItemStringComparator = new PriorityItemStringComparator();
		PriorityQueue<PriorityItemString> pq = new PriorityQueue<>(priorityItemStringComparator);
		for (Map.Entry<String, Set<Integer>> entry : productIdToReviewsIds.entrySet()) {
			String pid = entry.getKey();
			double pidScore = 0;
			for (Integer rid : entry.getValue()) {
				int helpfulnessDenominator = this.iReader.getReviewHelpfulnessDenominator(rid);
				int helpfulnessNumerator = this.iReader.getReviewHelpfulnessNumerator(rid);
				int score = this.iReader.getReviewScore(rid);
				double helpPart = helpfulnessNumerator / ((double) helpfulnessDenominator + helpfulnessAddend);
				double scorePart = (score - scoreAddend);
				pidScore += helpPart * scorePart;
			}
			if (pq.size() < k) {
				pq.add(new PriorityItemString(pid, pidScore));
			} else {
				if (pidScore < pq.peek().priority) {
					pq.poll();
					pq.add(new PriorityItemString(pid, pidScore));
				}
			}
		}
		for (int i = 1; i <= this.iReader.getNumberOfReviews(); i++) {
			if (pq.size() >= k) {
				break;
			}
			String productId = this.iReader.getProductId(i);
			if (!productIdToReviewsIds.containsKey(productId)) {
				pq.add(new PriorityItemString(productId, -100));
				productIdToReviewsIds.put(productId, null);
			}
		}
		ArrayList<String> list = new ArrayList<>(k);
		for (int i = 0; i < k; i++) {
			list.add(null);
		}
		for (int i = 0; i < k; i++) {
			list.set(k - i - 1, pq.poll().item);
		}
		return list;
	}

	private void updateProductIdToReviewsIds(Enumeration<Integer> result, Set<Integer> rids, Map<String, Set<Integer>> productIdToReviewsIds) {
		while (result.hasMoreElements()) {
			Integer rid = result.nextElement();
			if (rids.contains(rid)) {
				continue;
			}
			String productId = this.iReader.getProductId(rid);
			Set<Integer> prodIdRids = productIdToReviewsIds.computeIfAbsent(productId, (key) -> new HashSet<>());
			prodIdRids.add(rid);
			rids.add(rid);
		}
	}

	// Sparse Vectors

	private static class SparseVectorTF extends SparseVector {

		private final String mapToMethod;

		// mapToMethod in [n , l, a,b]
		SparseVectorTF(String mapToMethod) {
			super();
			this.mapToMethod = mapToMethod;
		}

		@Override
		public void addValue(String key, Double value) {
			switch (mapToMethod) {
				case "n":
					if (value == 0) {
						return;
					}
					super.addValue(key, value);
					break;
				case "l":
					if (value == 0) {
						return;
					}
					super.addValue(key, 1 + Math.log10(value));
					break;
				case "b":
					if (value == 0) {
						return;
					}
					super.addValue(key, 1d);
					break;
				case "a":
				default:
					throw new IllegalStateException("Unexpected value: " + mapToMethod);
			}
		}

	}

	private static class SparseVectorIDF extends SparseVector {
		private final String mapToMethod;
		private final double N;

		// mapToMethod in [n , t , s]
		SparseVectorIDF(String mapToMethod, double N) {
			super();
			this.N = N;
			this.mapToMethod = mapToMethod;
		}

		@Override
		public void addValue(String key, Double value) {

			switch (mapToMethod) {
				case "n":
					if (value == 0) {
						return;
					}
					super.addValue(key, 1d);
					break;
				case "t":
					if (value == 0) {
						return;
					}
					super.addValue(key, Math.log(N / value));
					break;
				case "s":
					super.addValue(key, Math.log(N / (1 + value)) + 1);
					break;
				default:
					throw new IllegalStateException("Unexpected value: " + mapToMethod);
			}
		}

	}

	private static class SparseVectorNormalized extends SparseVector {

		// mapToMethod in [n , c, l]
		SparseVectorNormalized(SparseVectorTF tf, SparseVectorIDF idf, String mapToMethod) {
			super(tf.inner(idf));
			this.normalize(mapToMethod);
		}

		private void normalize(String mapToMethod) {
			switch (mapToMethod) {
				case "n":
					applyDivByScale(1d);
					break;
				case "c":
					applyDivByScale(this.magnitude());
					break;
				case "l":
				default:
					throw new IllegalStateException("Unexpected value: " + mapToMethod);
			}

		}


	}

	// PQs

	private static class PriorityItemInt {
		private final Integer item;
		private final double priority;

		PriorityItemInt(Integer item, double priority) {
			this.item = item;
			this.priority = priority;
		}
	}

	private static class PriorityItemString {
		private final String item;
		private final double priority;

		PriorityItemString(String item, double priority) {
			this.item = item;
			this.priority = priority;
		}
	}

	private static class PriorityItemComparator implements Comparator<PriorityItemInt> {

		public int compare(PriorityItemInt o1, PriorityItemInt o2) {
			double diffPriority = o1.priority - o2.priority;
			if (diffPriority > 0) {
				return 1;
			} else if (diffPriority < 0) {
				return -1;
			} else {
				return o2.item - o1.item;
			}
		}

	}

	private static class PriorityItemStringComparator implements Comparator<PriorityItemString> {

		public int compare(PriorityItemString o1, PriorityItemString o2) {
			double diffPriority = o1.priority - o2.priority;
			if (diffPriority > 0) {
				return 1;
			} else if (diffPriority < 0) {
				return -1;
			} else {
				return 0;
			}
		}

	}

	// PQ Enumeration

	public static class PQEnumeration implements Enumeration<Integer> {


		private final PriorityQueue<PriorityItemInt> pq;

		public PQEnumeration(PriorityQueue<PriorityItemInt> pq) {
			this.pq = pq;
		}

		@Override
		public boolean hasMoreElements() {
			return this.pq.size() > 0;
		}

		@Override
		public Integer nextElement() {
			PriorityItemInt next = pq.poll();
			return next.item;
		}
	}

	public static class PQStringEnumeration implements Enumeration<String> {


		private final PriorityQueue<PriorityItemString> pq;

		public PQStringEnumeration(PriorityQueue<PriorityItemString> pq) {
			this.pq = pq;
		}

		@Override
		public boolean hasMoreElements() {
			return this.pq.size() > 0;
		}

		@Override
		public String nextElement() {
			PriorityItemString next = pq.poll();
			return next.item;
		}
	}


}
