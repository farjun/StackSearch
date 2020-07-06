package webdata.compression;


import webdata.utils.TextUtils;
import webdata.utils.Utils;

import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

public class KInKMinusOneFrontCoding {
	public static final int DEFAULT_BLOCK_SIZE = 4;
	public final int block_size;
	public String term;
	public List<Integer> lengths;
	public List<Integer> prefixes;
	public List<Integer> termPtr;
	public List<Integer> invertedIndexRId;
	public List<Integer> invertedIndexFreqInRid;

	public KInKMinusOneFrontCoding(int block_size) {
		this.block_size = block_size;
	}

	public KInKMinusOneFrontCoding() {
		this(DEFAULT_BLOCK_SIZE);
	}

	public int getBlockNum(int tokenSize) {
		int block_size = this.block_size;
		return (tokenSize + block_size - 1) / block_size;
	}

	public int getLengthsSize(int tokenSize) {
		int kMinusOne = this.block_size - 1;
		// complete blocks
		int size = (this.getBlockNum(tokenSize) - 1) * kMinusOne;
		// partial block
		int mod = tokenSize % this.block_size;
		if (mod == 0) {
			// this is a complete one
			size += kMinusOne;
		} else {
			size += mod;
		}
		return size;
	}

	public int getPrefixesSize(int tokenSize) {
		int kMinusOne = this.block_size - 1;
		// complete blocks
		int size = (this.getBlockNum(tokenSize) - 1) * kMinusOne;
		// partial block
		int mod = (tokenSize - 1) % this.block_size;
		if (mod == 0) {
			// this is an empty one
		} else {
			size += mod;
		}
		return size;
	}

	public void encode(SortedMap<String, SortedMap<Integer, Integer>> dict) {
		// TODO consider get  SortedMap<String, Map<Integer, Integer>> , and use TreeMap Here (memory reasons)
		term = "";
		lengths = new ArrayList<>();
		prefixes = new ArrayList<>();
		termPtr = new ArrayList<>();
		invertedIndexRId = new ArrayList<>();
		invertedIndexFreqInRid = new ArrayList<>();
		StringBuilder termStringBuilder = new StringBuilder();
		String first_word = null;
		String block_word = null;
		int mega_string_size = 0;
		int counter = 0;
		for (Map.Entry<String, SortedMap<Integer, Integer>> entry : dict.entrySet()) {
			String word = entry.getKey();
			if (counter % this.block_size == 0) {
				if (block_word != null) {
					termStringBuilder.append(block_word);
					mega_string_size += block_word.length();
				}
				first_word = word;
				block_word = word;
				termPtr.add(mega_string_size);
			} else {
				int prefix = TextUtils.getPrefix(first_word, word);
				block_word = block_word.concat(word.substring(prefix));
				prefixes.add(prefix);
			}
			if (counter % this.block_size != this.block_size - 1) {
				lengths.add(word.length());
			}
			SortedMap<Integer, Integer> invertedData = entry.getValue();
			for (Map.Entry<Integer, Integer> rIdFreqPair : invertedData.entrySet()) {
				int rId = rIdFreqPair.getKey();
				int freqInRId = rIdFreqPair.getValue();
				this.invertedIndexRId.add(rId);
				this.invertedIndexFreqInRid.add(freqInRId);
			}
			counter++;
		}
		if (block_word != null) {
			termStringBuilder.append(block_word);
		}
		term = termStringBuilder.toString();
	}

	public int getStringIdByString(String token, RandomAccessFile fileInputStream) {
		int low = 0;
		int numOfBlocks = termPtr.size();
		int high = numOfBlocks - 1;
		while (low <= high) {
			int mid = low + (high - low) / 2;
			String midString = getStringAt(fileInputStream, mid);
			if (midString.compareTo(token) > 0) {
				high = mid - 1;
			} else {
				String midHighString = null;
				if (mid + 1 != numOfBlocks) {
					midHighString = getStringAt(fileInputStream, mid + 1);
				}
				if (midHighString != null && midHighString.compareTo(token) <= 0) {
					low = mid + 1;
				} else {
					return this.findTokenOnBlock(fileInputStream, token, mid);
				}
			}
		}
		return -1;
	}

	private int findTokenOnBlock(RandomAccessFile fileInputStream, String token, int blockNum) {
		String firstWord = this.getStringAt(fileInputStream, blockNum);
		if (firstWord.equals(token)) {
			return blockNum * this.block_size;
		}
		int kMinusOne = this.block_size - 1;
		int base = blockNum * (kMinusOne);
		int wordInBlocks = Math.min(prefixes.size() - base, kMinusOne);
		int offset = firstWord.length();
		for (int i = 1; i <= wordInBlocks; i++) {
			int lengthToRead;
			int prefixSize = prefixes.get(base);
			if (i == kMinusOne) {
				if (blockNum == this.termPtr.size() - 1) {
					lengthToRead = 100;
				} else {
					lengthToRead = termPtr.get(blockNum + 1) - termPtr.get(blockNum) - offset;
				}
			} else {
				int totalWordLength = lengths.get(base + 1);
				lengthToRead = totalWordLength - prefixSize;
			}
			String subWord = this.getStringAt(fileInputStream, blockNum, lengthToRead, offset);
			String word = firstWord.substring(0, prefixSize).concat(subWord);
			int compare = word.compareTo(token);
			if (compare == 0) {
				return blockNum * this.block_size + i;
			} else if (compare > 0) {
				return -1;
			}
			offset += lengthToRead;
			base++;
		}
		return -1;
	}

	private String getStringAt(RandomAccessFile fileInputStream, int blockNum, int lengthToRead, int readOffset) {
		int offset = termPtr.get(blockNum) + readOffset;
		byte[] midBlockWord = Utils.readFromPositionRA(fileInputStream, lengthToRead, offset);
		String ret = new String(midBlockWord);
		ret = ret.replaceAll("\u0000", ""); // if we read too much (end of file), then it's removed.
		return ret;
	}

	private String getStringAt(RandomAccessFile fileInputStream, int blockNum) {
		return this.getStringAt(fileInputStream, blockNum, lengths.get(blockNum * (this.block_size - 1)), 0);
	}

	public String getStringByStringId(RandomAccessFile fileInputStream, int id) {
		int blockNum = Math.floorDiv(id, this.block_size);
		String firstWord = this.getStringAt(fileInputStream, blockNum);
		int kMinusOne = this.block_size - 1;
		int base = blockNum * (kMinusOne);
		int wordToSkip = id - (blockNum * this.block_size);
		if (wordToSkip == 0) {
			return firstWord;
		}
		int offset = firstWord.length();
		for (int i = 1; i <= wordToSkip; i++) {
			int lengthToRead;
			int prefixSize = prefixes.get(base);
			if (i == kMinusOne) {
				if (blockNum == this.termPtr.size() - 1) {
					lengthToRead = 100;
				} else {
					lengthToRead = termPtr.get(blockNum + 1) - termPtr.get(blockNum) - offset;
				}
			} else {
				int totalWordLength = lengths.get(base + 1);
				lengthToRead = totalWordLength - prefixSize;
			}
			if (i == wordToSkip) {
				String subWord = this.getStringAt(fileInputStream, blockNum, lengthToRead, offset);
				return firstWord.substring(0, prefixSize).concat(subWord);
			}
			offset += lengthToRead;
			base++;
		}
		return null;
	}


}
