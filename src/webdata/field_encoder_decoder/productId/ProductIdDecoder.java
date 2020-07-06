package webdata.field_encoder_decoder.productId;

import webdata.utils.Utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Paths;
import java.util.List;

public class ProductIdDecoder {

	private final String dir;
	private final int numReviews;
	private final int numOfPIds;
	private final ProductIdCompressor productIdCompressor;
	private final List<Integer> termPtr;
	private final List<Integer> lengths;
	private final List<Integer> prefixes;
	private final List<Integer> rIdToPid;
	private final List<Integer> invertedIndexPtr;
	private final List<Integer> tokenFrequency;

	public ProductIdDecoder(String dir, int numReviews, int numOfPIds) {
		this.dir = dir;
		this.numReviews = numReviews;
		this.numOfPIds = numOfPIds;
		this.productIdCompressor = new ProductIdCompressor();
		this.termPtr = decodeTermPtr();
		this.lengths = decodeLengths();
		this.prefixes = decodePrefixes();
		this.rIdToPid = decodeRIdToPid();
		this.invertedIndexPtr = decodeInvertedIndexPtr();
		this.tokenFrequency = this.decodeTokenFrequency();
	}

	private List<Integer> decodeRIdToPid() {
		String path = Paths.get(dir, ProductIdConstants.REVIEW_ID_TO_P_ID_F_NAME.value).toString();
		return productIdCompressor.decompressReviewsToPId(Utils.readFile(path), numReviews);
	}

	private List<Integer> decodeTermPtr() {
		String path = Paths.get(dir, ProductIdConstants.TERM_PTR_FILE_NAME.value).toString();
		return productIdCompressor.decompressTermPtr(Utils.readFile(path), numOfPIds);
	}

	private List<Integer> decodeLengths() {
		String path = Paths.get(dir, ProductIdConstants.WORD_LEN_FILE_NAME.value).toString();
		return productIdCompressor.decompressLengths(Utils.readFile(path), numOfPIds);
	}

	private List<Integer> decodePrefixes() {
		String path = Paths.get(dir, ProductIdConstants.PREFIX_FILE_NAME.value).toString();
		return productIdCompressor.decompressPrefixes(Utils.readFile(path), numOfPIds);
	}


	public String getProductIdByReviewId(int reviewId){
		int tokenId = this.rIdToPid.get(reviewId);
		RandomAccessFile input = this.openInput(ProductIdConstants.TERM_FILE_NAME);
		ProductIdCompressor.CompressedDictResult dictResult = new ProductIdCompressor.CompressedDictResult(
				lengths, prefixes, termPtr
		);
		String ret = productIdCompressor.getTokenByIndex(tokenId, input, dictResult);
		closeInput(input);
		return ret;
	}

	private int getTokenId(String token) {
		RandomAccessFile fileInputStream = this.openInput(ProductIdConstants.TERM_FILE_NAME);
		ProductIdCompressor.CompressedDictResult dictResult = new ProductIdCompressor.CompressedDictResult(
				lengths, prefixes, termPtr
		);
		int ret = productIdCompressor.findTokenIdInDict(token, fileInputStream, dictResult);
		closeInput(fileInputStream);
		return ret;
	}

	public List<Integer> getInvertedIndexList(String token) {
		RandomAccessFile fileInputStream = this.openInput(ProductIdConstants.INVERTED_INDEX_FILE_NAME);
		int tokenId = this.getTokenId(token);
		if (tokenId == -1) {
			return null;
		}
		int position = this.invertedIndexPtr.get(tokenId);
		int bytesToRead;
		if (tokenId != this.invertedIndexPtr.size() - 1) {
			bytesToRead = this.invertedIndexPtr.get(tokenId + 1) - position;
		} else {
			bytesToRead = 100;
		}
		byte[] bytes = Utils.readFromPositionRA(fileInputStream, bytesToRead, position);
		this.closeInput(fileInputStream);
		return this.productIdCompressor.decompressInvertedIndex(bytes,this.getTokenFrequencyByTokenID(tokenId));
	}

	private int getTokenFrequencyByTokenID(int tokenId) {
		if (tokenId == -1) {
			return 0;
		}
		return this.tokenFrequency.get(tokenId);
	}

	private List<Integer> decodeInvertedIndexPtr() {
		String path = Paths.get(dir, ProductIdConstants.INVERTED_INDEX_PTR_FILE_NAME.value).toString();
		return productIdCompressor.decodeInvertedIndexPtr(Utils.readFile(path), numOfPIds);
	}

	private List<Integer> decodeTokenFrequency() {
		String path = Paths.get(dir, ProductIdConstants.FREQ_FILE_NAME.value).toString();
		return productIdCompressor.decompressTokenFrequency(Utils.readFile(path), numOfPIds);
	}

	private void closeInputStream(FileInputStream fileInputStream) {
		try {
			fileInputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void closeInput(RandomAccessFile randomAccessFile) {
		try {
			randomAccessFile.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private FileInputStream openInputStream(ProductIdConstants fileName){
		String path = Paths.get(dir, fileName.value).toString();
		FileInputStream fileInputStream = null;
		try {
			fileInputStream = new FileInputStream(path);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return fileInputStream;
	}

	private RandomAccessFile openInput(ProductIdConstants fileName){
		String path = Paths.get(dir, fileName.value).toString();
		RandomAccessFile randomAccessFile = null;
		try {
			randomAccessFile = new RandomAccessFile(path,"r");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return randomAccessFile;
	}
}
