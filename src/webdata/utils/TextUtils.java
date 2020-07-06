package webdata.utils;

import webdata.compression.UnaryCoder;

import java.util.*;
import java.util.regex.Pattern;

/**
 * This is a utils files which oriented around Text.
 * Means Relevant to: Text and ProductId
 */
public abstract class TextUtils {

	private static Pattern pattern = Pattern.compile("[^a-zA-Z0-9]+");

	public static SortedMap<String, SortedMap<Integer, Integer>> convertMapToSortedMap(Map<String, Map<Integer, Integer>> mapOfMap) {
		SortedMap<String, SortedMap<Integer, Integer>> ret = new TreeMap<>();
		for (Map.Entry<String, Map<Integer, Integer>> entry : mapOfMap.entrySet()) {
			ret.put(entry.getKey(), new TreeMap<>(entry.getValue()));
		}
		return ret;
	}

	public static TextSplitResult splitTexts(List<String> strings) {
		List<Integer> numberOfTokensPerEntry = new ArrayList<>();
		Map<String, Map<Integer, Integer>> allTokens = new HashMap<>();
		int size = strings.size();
		for (int i = 0; i < size; i++) {
			String reviewText = strings.get(i);
			splitReviewText(i, reviewText, numberOfTokensPerEntry, allTokens);
		}
		return new TextSplitResult(allTokens, numberOfTokensPerEntry);
	}

	public static void splitReviewText(
			int reviewId, String reviewText,
			List<Integer> numberOfTokensPerEntry,
			Map<String, Map<Integer, Integer>> allTokens) {
		String[] tokens = pattern.split(reviewText);
		int startFrom;
		if (tokens.length > 0 && tokens[0].equals("")) {
			startFrom = 1;
		} else {
			startFrom = 0;
		}
		if (numberOfTokensPerEntry != null){
			numberOfTokensPerEntry.add(tokens.length - startFrom);
		}
		for (; startFrom < tokens.length; startFrom++) {
			String token = tokens[startFrom];
			Map<Integer, Integer> tokenMap = allTokens.computeIfAbsent(token, (String key) -> new HashMap<>());
			tokenMap.put(reviewId, tokenMap.computeIfAbsent(reviewId, (Integer key) -> 0) + 1);
		}
	}

	public static TokensFrequenciesResult getTokensFrequencies(SortedMap<String, SortedMap<Integer, Integer>> sortedMap) {
		List<Integer> tokenFrequency = new ArrayList<>();
		List<Integer> tokenCollectionFrequency = new ArrayList<>();
		for (Map.Entry<String, SortedMap<Integer, Integer>> entry : sortedMap.entrySet()) {
			SortedMap<Integer, Integer> r_id_to_freq = entry.getValue();
			tokenFrequency.add(r_id_to_freq.size());
			Integer sum = r_id_to_freq.values().stream().reduce(0, Integer::sum);
			tokenCollectionFrequency.add(sum);
		}
		return new TokensFrequenciesResult(tokenFrequency, tokenCollectionFrequency);
	}

	public static List<List<Integer>> invertedRIds(
			List<Integer> tokenFrequency,
			List<Integer> invertedIndexRIds,
			List<Integer> invertedIndexFreqInRId
	) {
		int freqSum = 0;
		int curFreq = 0;
		List<List<Integer>> listOfList = new ArrayList<>();
		for (int freq : tokenFrequency) {
			int lastId = 0;
			ArrayList<Integer> midList = new ArrayList<>();
			freqSum += freq;
			for (; curFreq < freqSum; curFreq++) {
				Integer id = invertedIndexRIds.get(curFreq);
				midList.add(id - lastId + 1); // Add The +1 for the "real" id.
				lastId = id + 1;
				if (invertedIndexFreqInRId != null) {
					midList.add(invertedIndexFreqInRId.get(curFreq));
				}
			}
			listOfList.add(midList);
		}
		return listOfList;
	}

	public static List<List<Integer>> invertedRIds(
			List<Integer> tokenFrequency,
			List<Integer> invertedIndexRIds
	) {
		return invertedRIds(tokenFrequency, invertedIndexRIds, null);
	}

	public static InvertedIndexEncodeResult encodeLists(
			UnaryCoder coder,
			List<List<Integer>> invertedRIds) {
		List<Byte> result = new ArrayList<>();
		List<Integer> invertedIndexPtr = new ArrayList<>();
		int termPtr = 0;
		for (List<Integer> invertedRId : invertedRIds) {
			invertedIndexPtr.add(termPtr);
			List<Byte> encode = coder.encode(invertedRId);
			result.addAll(encode);
			termPtr += encode.size();
		}
		return new InvertedIndexEncodeResult(result, invertedIndexPtr);
	}

	public static int getPrefix(String first_word, String word) {
		int prefix;
		int max_size = Math.min(first_word.length(), word.length());
		for (prefix = 0; prefix < max_size; prefix++) {
			if (first_word.charAt(prefix) != word.charAt(prefix)) {
				break;
			}
		}
		return prefix;
	}

	public static class InvertedIndexEncodeResult {

		public final List<Byte> result;
		public final List<Integer> invertedIndexPtr;

		public InvertedIndexEncodeResult(List<Byte> result, List<Integer> invertedIndexPtr) {

			this.result = result;
			this.invertedIndexPtr = invertedIndexPtr;
		}
	}

	public static class TokensFrequenciesResult {

		public final List<Integer> tokenFrequency;
		public final List<Integer> tokenCollectionFrequency;

		public TokensFrequenciesResult(List<Integer> tokenFrequency, List<Integer> tokenCollectionFrequency) {

			this.tokenFrequency = tokenFrequency;
			this.tokenCollectionFrequency = tokenCollectionFrequency;
		}
	}

	public static class TextSplitResult {

		public final Map<String, Map<Integer, Integer>> allTokens;
		public final List<Integer> numberOfTokensPerEntry;

		public TextSplitResult(Map<String, Map<Integer, Integer>> allTokens, List<Integer> numberOfToken) {
			this.allTokens = allTokens;
			this.numberOfTokensPerEntry = numberOfToken;
		}
	}


}
