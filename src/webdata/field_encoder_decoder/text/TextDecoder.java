package webdata.field_encoder_decoder.text;

import webdata.utils.Utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Paths;
import java.util.List;

public class TextDecoder {

	private final String dir;
	private final int numReviews;
	private final int numOfTokens;

	private final List<Integer> reviewLengths;
	private final List<Integer> tokenFrequency;
	private final List<Integer> tokenCollectionFrequency;
	private final List<Integer> termPtr;
	private final List<Integer> lengths;
	private final List<Integer> prefixes;
	private final List<Integer> invertedIndexPtr;
	private final TextCompressor textCompressor;

	public TextDecoder(String dir, int numReviews, int numOfTokens) {
		this.dir = dir;
		this.numReviews = numReviews;
		this.numOfTokens = numOfTokens;
		textCompressor = new TextCompressor();
		this.reviewLengths = this.decodeReviewLengths();
		this.tokenFrequency = this.decodeTokenFrequency();
		this.tokenCollectionFrequency = this.decodeTokenCollectionFrequency();
		this.invertedIndexPtr = this.decodeInvertedIndexPtr();
		this.termPtr = decodeTermPtr();
		this.lengths = decodeLengths();
		this.prefixes = decodePrefixes();
	}

	private List<Integer> decodeTermPtr() {
		String path = Paths.get(dir, TextConstants.TERM_PTR_FILE_NAME.value).toString();
		return textCompressor.decompressTermPtr(Utils.readFile(path), numOfTokens);
	}

	private List<Integer> decodeLengths() {
		String path = Paths.get(dir, TextConstants.WORD_LEN_FILE_NAME.value).toString();
		return textCompressor.decompressLengths(Utils.readFile(path), numOfTokens);
	}

	private List<Integer> decodePrefixes() {
		String path = Paths.get(dir, TextConstants.PREFIX_FILE_NAME.value).toString();
		return textCompressor.decompressPrefixes(Utils.readFile(path), numOfTokens);
	}


	public int getReviewLengthByReviewID(int reviewId) {
		return this.reviewLengths.get(reviewId);
	}

	private int getTokenCollectionFrequencyByTokenID(int tokenId) {
		if (tokenId == -1) {
			return 0;
		}
		return this.tokenCollectionFrequency.get(tokenId);
	}

	private int getTokenFrequencyByTokenID(int tokenId) {
		if (tokenId == -1) {
			return 0;
		}
		return this.tokenFrequency.get(tokenId);
	}

	public int getTokenCollectionFrequencyByToken(String token) {
		return this.getTokenCollectionFrequencyByTokenID(getTokenId(token));
	}

	public int getTokenFrequencyByToken(String token) {
		return this.getTokenFrequencyByTokenID(getTokenId(token));
	}

	public int getTokenSizeOfReviews() {
		int sum = 0;
		for (Integer freqInRID : this.tokenCollectionFrequency) {
			sum += freqInRID;
		}
		return sum;
	}

	private int getTokenId(String token) {
		RandomAccessFile fileInputStream = this.openInput(TextConstants.TERM_FILE_NAME);
		TextCompressor.CompressedDictResult dictResult = new TextCompressor.CompressedDictResult(
				lengths, prefixes, termPtr
		);
		int ret = textCompressor.findTokenIdInDict(token, fileInputStream, dictResult);
		this.closeInput(fileInputStream);
		return ret;
	}

	public List<Integer> getInvertedIndexList(String token) {
		int tokenId = this.getTokenId(token);
		if (tokenId == -1) {
			return null;
		}
		RandomAccessFile fileInputStream = this.openInput(TextConstants.INVERTED_INDEX_FILE_NAME);
		int position = this.invertedIndexPtr.get(tokenId);
		int bytesToRead;
		if (tokenId != this.invertedIndexPtr.size() - 1) {
			bytesToRead = this.invertedIndexPtr.get(tokenId + 1) - position;
		} else {
			bytesToRead = 100;
		}
		byte[] bytes = Utils.readFromPositionRA(fileInputStream, bytesToRead, position);
		this.closeInput(fileInputStream);
		return this.textCompressor.decompressInvertedIndex(bytes, this.getTokenFrequencyByTokenID(tokenId));
	}


	private List<Integer> decodeTokenCollectionFrequency() {
		String path = Paths.get(dir, TextConstants.COLLECTION_FREQ_FILE_NAME.value).toString();
		return textCompressor.decompressCollectionFrequency(Utils.readFile(path), numOfTokens);
	}

	private List<Integer> decodeInvertedIndexPtr() {
		String path = Paths.get(dir, TextConstants.INVERTED_INDEX_PTR_FILE_NAME.value).toString();
		return textCompressor.decodeInvertedIndexPtr(Utils.readFile(path), numOfTokens);
	}

	private List<Integer> decodeTokenFrequency() {
		String path = Paths.get(dir, TextConstants.FREQ_FILE_NAME.value).toString();
		return textCompressor.decompressTokenFrequency(Utils.readFile(path), numOfTokens);
	}

	private List<Integer> decodeReviewLengths() {
		String path = Paths.get(dir, TextConstants.REVIEW_LEN_FILE_NAME.value).toString();
		return textCompressor.decompressReviewLength(Utils.readFile(path), numReviews);
	}

// TODO remove, used for inverted.
//	private void closeInputStream(FileInputStream fileInputStream) {
//		try {
//			fileInputStream.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}

//	private FileInputStream openInputStream(TextConstants fileName) {
//		String path = Paths.get(dir, fileName.value).toString();
//		FileInputStream fileInputStream = null;
//		try {
//			fileInputStream = new FileInputStream(path);
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
//		return fileInputStream;
//
//	}

	private void closeInput(RandomAccessFile fileInputStream) {
		try {
			fileInputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private RandomAccessFile openInput(TextConstants fileName) {
		String path = Paths.get(dir, fileName.value).toString();
		RandomAccessFile fileInputStream = null;
		try {
			fileInputStream = new RandomAccessFile(path, "r");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return fileInputStream;

	}
}
