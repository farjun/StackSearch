package webdata.field_encoder_decoder.text;

import webdata.field_encoder_decoder.Encoder;
import webdata.field_encoder_decoder.meta.MetaEncoder;
import webdata.utils.TextUtils;
import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.*;

public class TextEncoder implements Encoder {

	private String dir;
	private Map<String, Map<Integer, Integer>> allTokens;
	private List<Integer> reviewIdToReviewSize;

	public TextEncoder(String dir) {
		this.reset(dir);
	}

	public void encode(List<String> texts) {
		TextUtils.TextSplitResult textSplitResult = TextUtils.splitTexts(texts);
		allTokens = textSplitResult.allTokens;
		reviewIdToReviewSize = textSplitResult.numberOfTokensPerEntry;
		MetaEncoder.getInstance().setTokenSize(allTokens.size());
		this.saveResult();

	}

	private void writeListsToFile(
			List<Integer> reviewIdToReviewSize,
			List<Integer> tokenFrequency,
			List<Integer> tokenCollectionFrequency,
			TextCompressor textCompressor) {
		writeToFile(TextConstants.REVIEW_LEN_FILE_NAME,
				textCompressor.compressReviewLength(reviewIdToReviewSize));
		writeToFile(TextConstants.COLLECTION_FREQ_FILE_NAME,
				textCompressor.compressTokenCollectionFrequency(tokenCollectionFrequency));
		writeToFile(TextConstants.FREQ_FILE_NAME,
				textCompressor.compressTokenFrequency(tokenFrequency));
	}

	private void writeDict(SortedMap<String, SortedMap<Integer, Integer>> sortedMap, List<Integer> tokenFrequency, TextCompressor textCompressor) {
		TextCompressor.CompressedDictResult dictResult = textCompressor.compressDict(sortedMap);
		writeToFile(TextConstants.TERM_PTR_FILE_NAME,
				textCompressor.compressTermPtr(dictResult.termPtr));
		writeToFile(TextConstants.WORD_LEN_FILE_NAME,
				textCompressor.compressLengths(dictResult.lengths));
		writeToFile(TextConstants.PREFIX_FILE_NAME,
				textCompressor.compressPrefixes(dictResult.prefixes));
		writeToFile(TextConstants.TERM_FILE_NAME,
				dictResult.term.getBytes());
		TextCompressor.InvertedCompressResult invertedCompressResult =
				textCompressor.compressInvertedIndex(
						tokenFrequency,
						dictResult.invertedIndexRId,
						dictResult.invertedIndexFreqInRId
				);
		writeToFile(TextConstants.INVERTED_INDEX_FILE_NAME, invertedCompressResult.result);
		writeToFile(TextConstants.INVERTED_INDEX_PTR_FILE_NAME, invertedCompressResult.invertedIndexPtr);
	}

	private void writeToFile(TextConstants fName, List<Byte> compressed) {
		String path = Paths.get(dir, fName.value).toString();
		Utils.writeFile(path, compressed);
	}

	private void writeToFile(TextConstants fName, byte[] compressed) {
		String path = Paths.get(dir, fName.value).toString();
		Utils.writeFile(path, compressed);
	}


	public void add(int reviewId, String text) {
		TextUtils.splitReviewText(reviewId, text, reviewIdToReviewSize, allTokens);
	}

	public void saveResult() {
		MetaEncoder.getInstance().setTokenSize(allTokens.size());
		SortedMap<String, SortedMap<Integer, Integer>> sortedMap = TextUtils.convertMapToSortedMap(allTokens);
		allTokens = new HashMap<>();
//		ArrayList<String> strings = new ArrayList<>(sortedMap.keySet());
//		Random random = new Random();
//		System.out.print("[");
//		for (int i = 0; i < 100; i++) {
//			System.out.printf("\"%s\",", strings.get(random.nextInt(strings.size())));
//		}
//		System.out.print("]");
		TextUtils.TokensFrequenciesResult tokensFrequenciesResult = TextUtils.getTokensFrequencies(sortedMap);
		List<Integer> tokenFrequency = tokensFrequenciesResult.tokenFrequency;
		List<Integer> tokenCollectionFrequency = tokensFrequenciesResult.tokenCollectionFrequency;
		TextCompressor textCompressor = new TextCompressor();
		writeListsToFile(reviewIdToReviewSize, tokenFrequency, tokenCollectionFrequency, textCompressor);
		writeDict(sortedMap, tokenFrequency, textCompressor);
	}

	public void reset(String dir) {
		this.dir = dir;
		reviewIdToReviewSize = new ArrayList<>();
		allTokens = new HashMap<>();
	}
}
