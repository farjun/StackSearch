package webdata.field_encoder_decoder.productId;

import webdata.field_encoder_decoder.Encoder;
import webdata.field_encoder_decoder.meta.MetaEncoder;
import webdata.utils.TextUtils;
import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.*;

public class ProductIdEncoder implements Encoder {
	private String dir;
	private Map<String, Map<Integer, Integer>> allTokens;
	private int firstRID;

	public ProductIdEncoder(String dir) {
		this.reset(dir);
	}

	public void encode(List<String> productIds) {
		TextUtils.TextSplitResult textSplitResult = TextUtils.splitTexts(productIds);
		this.firstRID = 0;
		allTokens = textSplitResult.allTokens;
		saveResult();
	}

	private void writeListToFile(List<Integer> tokenFrequency, ProductIdCompressor productIdCompressor) {
		writeToFile(ProductIdConstants.FREQ_FILE_NAME, productIdCompressor.compressTokenFrequency(tokenFrequency));
	}

	private void writeDictToFile(SortedMap<String, SortedMap<Integer, Integer>> sortedMap, List<Integer> tokenFrequency, ProductIdCompressor productIdCompressor) {
		ProductIdCompressor.CompressedDictResult dictResult = productIdCompressor.compressDict(sortedMap);
		writeToFile(ProductIdConstants.TERM_PTR_FILE_NAME,
				productIdCompressor.compressTermPtr(dictResult.termPtr));
		writeToFile(ProductIdConstants.WORD_LEN_FILE_NAME,
				productIdCompressor.compressLengths(dictResult.lengths));
		writeToFile(ProductIdConstants.PREFIX_FILE_NAME,
				productIdCompressor.compressPrefixes(dictResult.prefixes));
		writeToFile(ProductIdConstants.TERM_FILE_NAME,
				dictResult.term.getBytes());
		ProductIdCompressor.InvertedCompressResult invertedCompressResult = productIdCompressor.compressInvertedIndex(tokenFrequency, dictResult.invertedIndexRId);
		writeToFile(ProductIdConstants.INVERTED_INDEX_FILE_NAME, invertedCompressResult.result);
		writeToFile(ProductIdConstants.INVERTED_INDEX_PTR_FILE_NAME, invertedCompressResult.invertedIndexPtr);
	}

	private void writeRIdToPId(SortedMap<String, SortedMap<Integer, Integer>> sortedMap, ProductIdCompressor productIdCompressor) {
		Set<SortedMap.Entry<String, SortedMap<Integer, Integer>>> entries = sortedMap.entrySet();
		List<Integer> rIdToPId = new ArrayList<>();
		int pId = 0;
		int highRID = this.firstRID - 1;
		for (Map.Entry<String, SortedMap<Integer, Integer>> entry : entries) {
			Set<Integer> rIds = entry.getValue().keySet();
			for (int rId : rIds) {
				if (rId > highRID) {
					for (int i = 0; i < rId - highRID; i++) {
						rIdToPId.add(0);
					}
					highRID = rId;
				}
				rIdToPId.set(rId - this.firstRID, pId);
			}
			pId++;
		}
		writeToFile(ProductIdConstants.REVIEW_ID_TO_P_ID_F_NAME,
				productIdCompressor.compressReviewsToPId(rIdToPId));
	}

	private void writeToFile(ProductIdConstants fName, List<Byte> compressed) {
		String path = Paths.get(dir, fName.value).toString();
		Utils.writeFile(path, compressed);
	}

	private void writeToFile(ProductIdConstants fName, byte[] compressed) {
		String path = Paths.get(dir, fName.value).toString();
		Utils.writeFile(path, compressed);
	}

	public void add(int reviewId, String productId) {
		if (firstRID == -1) {
			firstRID = reviewId;
		}
		TextUtils.splitReviewText(reviewId, productId, null, allTokens);
	}

	public void saveResult() {
		MetaEncoder.getInstance().setPIdSize(allTokens.size());
		SortedMap<String, SortedMap<Integer, Integer>> sortedMap = TextUtils.convertMapToSortedMap(allTokens);
		allTokens = new HashMap<>();
		TextUtils.TokensFrequenciesResult tokensFrequenciesResult = TextUtils.getTokensFrequencies(sortedMap);
		List<Integer> tokenFrequency = tokensFrequenciesResult.tokenFrequency;
		ProductIdCompressor productIdCompressor = new ProductIdCompressor();
		writeListToFile(tokenFrequency, productIdCompressor);
		writeDictToFile(sortedMap, tokenFrequency, productIdCompressor);
		writeRIdToPId(sortedMap, productIdCompressor);
	}

	public void reset(String dir) {
		this.dir = dir;
		allTokens = new HashMap<>();
		firstRID = -1;
	}
}
