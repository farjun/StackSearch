package webdata.field_encoder_decoder.productId;

import webdata.compression.*;
import webdata.field_encoder_decoder.text.WordInfo;
import webdata.utils.TextUtils;
import webdata.utils.Utils;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.SortedMap;
import java.util.stream.Collectors;

public class ProductIdCompressor {

	private final KInKMinusOneFrontCoding dictCoder;
	private final UnaryCoder termPtrCoder;
	private final UnaryCoder lengthsCoder;
	private final UnaryCoder prefixesCoder;
	private final UnaryCoder invertedIndexPtrCoder;
	private final UnaryCoder tokenFrequencyCoder;
	private final UnaryCoder reviewIdToPIdCoder;
	private final UnaryCoder invertedIndexCoder;

	ProductIdCompressor() {
		this.tokenFrequencyCoder = new GammaCode();
		this.dictCoder = new KInKMinusOneFrontCoding();
		this.termPtrCoder = new DeltaCodeGap(1);
		this.lengthsCoder = new DeltaCode();
		this.prefixesCoder = new GammaCode(1);
		this.reviewIdToPIdCoder = new DeltaCode(1);
		this.invertedIndexCoder = new GammaCode();
		this.invertedIndexPtrCoder = new DeltaCodeGap(1);
	}

	public List<Byte> compressTermPtr(List<Integer> termPtr) {
		return this.termPtrCoder.encode(termPtr);
	}

	public List<Integer> decompressTermPtr(byte[] bytes, int tokenSize) {
		int size = this.dictCoder.getBlockNum(tokenSize);
		return this.termPtrCoder.decode(size, bytes);
	}


	public List<Byte> compressLengths(List<Integer> lengths) {
		return this.lengthsCoder.encode(lengths);
	}

	public List<Integer> decompressLengths(byte[] bytes, int tokenSize) {
		int size = this.dictCoder.getLengthsSize(tokenSize);
		return this.lengthsCoder.decode(size, bytes);
	}


	public List<Byte> compressPrefixes(List<Integer> prefixes) {
		return this.prefixesCoder.encode(prefixes);
	}

	public List<Integer> decompressPrefixes(byte[] bytes, int tokenSize) {
		int size = this.dictCoder.getPrefixesSize(tokenSize);
		return this.prefixesCoder.decode(size, bytes);
	}


	public CompressedDictResult compressDict(SortedMap<String, SortedMap<Integer, Integer>> dict) {
		this.dictCoder.encode(dict);
		return new CompressedDictResult(
				this.dictCoder.lengths,
				this.dictCoder.prefixes,
				this.dictCoder.termPtr,
				this.dictCoder.invertedIndexRId,
				this.dictCoder.invertedIndexFreqInRid,
				this.dictCoder.term
		);
	}

	public int findTokenIdInDict(String token,
								 RandomAccessFile fileInputStream,
								 CompressedDictResult compressedDictResult) {
		this.dictCoder.lengths = compressedDictResult.lengths;
		this.dictCoder.prefixes = compressedDictResult.prefixes;
		this.dictCoder.termPtr = compressedDictResult.termPtr;
		return this.dictCoder.getStringIdByString(token, fileInputStream);
	}

	public String getTokenByIndex(int id,
								  RandomAccessFile fileInputStream,
								  CompressedDictResult compressedDictResult) {
		this.dictCoder.lengths = compressedDictResult.lengths;
		this.dictCoder.prefixes = compressedDictResult.prefixes;
		this.dictCoder.termPtr = compressedDictResult.termPtr;
		return this.dictCoder.getStringByStringId(fileInputStream, id);
	}

	public List<Byte> compressReviewsToPId(List<Integer> rIdToPId) {
		return reviewIdToPIdCoder.encode(rIdToPId);
	}

	public List<Integer> decompressReviewsToPId(byte[] bytes, int size) {
		return reviewIdToPIdCoder.decode(size, bytes);
	}

	public InvertedCompressResult compressInvertedIndex(
			List<Integer> tokenFrequency,
			List<Integer> invertedIndexRIds) {
		List<List<Integer>> invertedRIds = TextUtils.invertedRIds(tokenFrequency, invertedIndexRIds);
		TextUtils.InvertedIndexEncodeResult encodeResult = TextUtils.encodeLists(
				this.invertedIndexCoder,
				invertedRIds
		);
		return new InvertedCompressResult(encodeResult.result,
				this.invertedIndexPtrCoder.encode(encodeResult.invertedIndexPtr));
	}

	public List<Integer> decodeInvertedIndexPtr(byte[] data, int size) {
		return this.invertedIndexPtrCoder.decode(size, data);
	}

	public List<Byte> compressTokenFrequency(List<Integer> integerList) {
		return this.tokenFrequencyCoder.encode(integerList);
	}

	public List<Integer> decompressTokenFrequency(byte[] bytes, int size) {
		return this.tokenFrequencyCoder.decode(size, bytes);
	}

	public List<Integer> decompressInvertedIndex(byte[] data, int size) {
		List<Integer> decode = this.invertedIndexCoder.decode(size, data);
		Utils.addLastValueForEachM(decode, 1);
		return decode;
	}

	public void mergeRIdToPid(InputStreamWrapper wrapper1, int amountOfReviews1,
							  InputStreamWrapper wrapper2, int amountOfReviews2,
							  OutputStreamWrapper outputStreamWrapper) {
		this.reviewIdToPIdCoder.mergeTwoInputToOutput(
				amountOfReviews1, amountOfReviews2,
				wrapper1.rIdToPIdInputStream,
				wrapper2.rIdToPIdInputStream,
				outputStreamWrapper.rIdToPIdOutputStream
		);
	}

	public int mergeInputsToOutput(InputStreamWrapper wrapper1,
								   int amountOfReviews1, InputStreamWrapper wrapper2,
								   int amountOfReviews2, OutputStreamWrapper outputStreamWrapper) {
		WordInfoIter iter1 = new WordInfoIter(wrapper1);
		WordInfoIter iter2 = new WordInfoIter(wrapper2);
		WordInfo info1 = iter1.next();
		WordInfo info2 = iter2.next();
		MergerKInKMinusOneFrontCoding dictMerger = new MergerKInKMinusOneFrontCoding(this.dictCoder.block_size);
		InfoWriter writer = new InfoWriter(outputStreamWrapper);
		int amountOfTokens = 0;
		List<Integer> map1 = new ArrayList<>();
		List<Integer> map2 = new ArrayList<>();
		while (info1 != null && info2 != null) {
			int cmp = info1.word.compareTo(info2.word);
			if (cmp < 0) {
				addInfo(dictMerger, writer, info1.word, info1.freq, info1.collectionFreq, info1.invertedIndex);
				info1 = iter1.next();
				map1.add(amountOfTokens);
			} else if (cmp > 0) {
				addInfo(dictMerger, writer, info2.word, info2.freq, info2.collectionFreq, info2.invertedIndex);
				info2 = iter2.next();
				map2.add(amountOfTokens);
			} else { // same
				info1.invertedIndex.addAll(info2.invertedIndex);
				addInfo(dictMerger, writer, info1.word,
						info1.freq + info2.freq,
						info1.collectionFreq + info2.collectionFreq,
						info1.invertedIndex
				);
				info1 = iter1.next();
				info2 = iter2.next();
				map1.add(amountOfTokens);
				map2.add(amountOfTokens);
			}
			amountOfTokens++;
		}
		while (info1 != null) {
			addInfo(dictMerger, writer, info1.word, info1.freq, info1.collectionFreq, info1.invertedIndex);
			info1 = iter1.next();
			map1.add(amountOfTokens);
			amountOfTokens++;
		}
		while (info2 != null) {
			addInfo(dictMerger, writer, info2.word, info2.freq, info2.collectionFreq, info2.invertedIndex);
			info2 = iter2.next();
			map2.add(amountOfTokens);
			amountOfTokens++;
		}
		iter1.terminate();
		iter2.terminate();
		writer.writeFinalResult();
		this.reviewIdToPIdCoder.mergeTwoInputToOutputWithMap(
				amountOfReviews1, amountOfReviews2,
				wrapper1.rIdToPIdInputStream, wrapper2.rIdToPIdInputStream,
				map1, map2,
				outputStreamWrapper.rIdToPIdOutputStream
		);
		return amountOfTokens;
	}


	public int megaMerge(List<InputStreamWrapper> wrappers, List<Integer> amountOfReviews, OutputStreamWrapper outputStreamWrapper) {

		int size = wrappers.size();
		List<WordInfoIter> wordInfoIters = new ArrayList<>(size);
		for (InputStreamWrapper inputStreamWrapper : wrappers) {
			wordInfoIters.add(new WordInfoIter(inputStreamWrapper));
		}
		MergerKInKMinusOneFrontCoding dictMerger = new MergerKInKMinusOneFrontCoding(this.dictCoder.block_size);
		InfoWriter writer = new InfoWriter(outputStreamWrapper);
		int amountOfTokens = 0;
		List<WordInfo> nexts = new ArrayList<>(size);
		for (WordInfoIter iter : wordInfoIters) {
			nexts.add(iter.next());
		}
		List<List<Integer>> maps = new ArrayList<>();
		for (Integer amountOfReview : amountOfReviews) {
			maps.add(new ArrayList<>(amountOfReview));
		}

		List<Integer> argmin = new ArrayList<>();
		WordInfo min = null;
		int notNull = nexts.size();
		while (notNull > 0) {
			for (int i = 0; i < size; i++) {
				WordInfo wordInfo = nexts.get(i);
				int cmp;
				if (wordInfo == null) {
					continue;
				} else if (min == null) {
					cmp = -1;
				} else {
					cmp = wordInfo.word.compareTo(min.word);
				}
				if (cmp == 0) {
					argmin.add(i);
				} else if (cmp < 0) {
					argmin.clear();
					min = wordInfo;
					argmin.add(i);
				}
			}
			int freq = 0;
			int collectionFreq = 0;
			List<List<Integer>> manyInverted = new ArrayList<>();
			for (int index : argmin) {
				WordInfo info = nexts.get(index);
				freq += info.freq;
				collectionFreq += info.collectionFreq;
				manyInverted.add(info.invertedIndex);
				WordInfo next = wordInfoIters.get(index).next();
				if (next == null) {
					notNull--;
				}
				nexts.set(index, next);
				maps.get(index).add(amountOfTokens);
			}
			this.addInfoWithManyInverted(dictMerger, writer, min.word, freq, collectionFreq, manyInverted);
			argmin.clear();
			min = null;
			amountOfTokens++;
		}
		for (WordInfoIter iter : wordInfoIters) {
			iter.terminate();
		}
		this.reviewIdToPIdCoder.mergeInputToOutputWithMaps(
				amountOfReviews,
				wrappers.stream().map((w) -> w.rIdToPIdInputStream).collect(Collectors.toList()),
				maps,
				outputStreamWrapper.rIdToPIdOutputStream
		);
		writer.writeFinalResult();
		return amountOfTokens;


	}

	private void addInfoWithManyInverted(
			MergerKInKMinusOneFrontCoding dictMerger,
			InfoWriter writer,
			String word,
			int freq,
			int collectionFreq,
			List<List<Integer>> invertedIndexes
	) {
		MergerKInKMinusOneFrontCoding.WriteInfo info = dictMerger.addWord(word);
		writer.writeInfoToFile(info, freq);
		writer.writeInvertedIndexes(invertedIndexes);
	}


	private void addInfo(MergerKInKMinusOneFrontCoding dictMerger,
						 InfoWriter writer,
						 String word,
						 int freq,
						 int collectionFreq,
						 List<Integer> invertedIndex
	) {
		int lastValue = 0;
		for (int i = 0; i < invertedIndex.size(); i += 1) {
			Integer value = invertedIndex.get(i);
			invertedIndex.set(i, value - lastValue);
			lastValue = value;
		}
		MergerKInKMinusOneFrontCoding.WriteInfo info = dictMerger.addWord(word);
		writer.writeInfoToFile(info, freq);
	}


	public static class InputStreamWrapper {

		private final InputStream termInputStream;
		private final InputStream prefixInputStream;
		private final InputStream lengthInputStream;
		private final InputStream termPtrInputStream;
		private final InputStream freqInputStream;
		private final InputStream invertedIndexInputStream;
		private final InputStream invertedIndexPtrInputStream;
		private final InputStream rIdToPIdInputStream;

		InputStreamWrapper(String dir) throws FileNotFoundException {
			String f1 = Paths.get(dir, ProductIdConstants.TERM_FILE_NAME.value).toString();
			String f2 = Paths.get(dir, ProductIdConstants.PREFIX_FILE_NAME.value).toString();
			String f3 = Paths.get(dir, ProductIdConstants.WORD_LEN_FILE_NAME.value).toString();
			String f4 = Paths.get(dir, ProductIdConstants.TERM_PTR_FILE_NAME.value).toString();
			String f5 = Paths.get(dir, ProductIdConstants.REVIEW_ID_TO_P_ID_F_NAME.value).toString();
			String f6 = Paths.get(dir, ProductIdConstants.FREQ_FILE_NAME.value).toString();
			String f8 = Paths.get(dir, ProductIdConstants.INVERTED_INDEX_FILE_NAME.value).toString();
			String f9 = Paths.get(dir, ProductIdConstants.INVERTED_INDEX_PTR_FILE_NAME.value).toString();

			termInputStream = new FileInputStream(f1);
			prefixInputStream = new FileInputStream(f2);
			lengthInputStream = new FileInputStream(f3);
			termPtrInputStream = new FileInputStream(f4);
			rIdToPIdInputStream = new FileInputStream(f5);
			freqInputStream = new FileInputStream(f6);
			invertedIndexInputStream = new FileInputStream(f8);
			invertedIndexPtrInputStream = new FileInputStream(f9);
		}

		void close() throws IOException {
			termInputStream.close();
			prefixInputStream.close();
			lengthInputStream.close();
			termPtrInputStream.close();
			freqInputStream.close();
			rIdToPIdInputStream.close();
			invertedIndexInputStream.close();
			invertedIndexPtrInputStream.close();
		}
	}

	public static class OutputStreamWrapper {

		private final OutputStream termOutputStream;
		private final OutputStream prefixOutputStream;
		private final OutputStream lengthOutputStream;
		private final OutputStream termPtrOutputStream;
		private final OutputStream freqOutputStream;
		private final OutputStream invertedIndexOutputStream;
		private final OutputStream invertedIndexPtrOutputStream;
		private final OutputStream rIdToPIdOutputStream;


		OutputStreamWrapper(String dir) throws FileNotFoundException {
			String f1 = Paths.get(dir, ProductIdConstants.TERM_FILE_NAME.value).toString();
			String f2 = Paths.get(dir, ProductIdConstants.PREFIX_FILE_NAME.value).toString();
			String f3 = Paths.get(dir, ProductIdConstants.WORD_LEN_FILE_NAME.value).toString();
			String f4 = Paths.get(dir, ProductIdConstants.TERM_PTR_FILE_NAME.value).toString();
			String f5 = Paths.get(dir, ProductIdConstants.REVIEW_ID_TO_P_ID_F_NAME.value).toString();
			String f6 = Paths.get(dir, ProductIdConstants.FREQ_FILE_NAME.value).toString();
			String f8 = Paths.get(dir, ProductIdConstants.INVERTED_INDEX_FILE_NAME.value).toString();
			String f9 = Paths.get(dir, ProductIdConstants.INVERTED_INDEX_PTR_FILE_NAME.value).toString();

			termOutputStream = new FileOutputStream(f1);
			prefixOutputStream = new FileOutputStream(f2);
			lengthOutputStream = new FileOutputStream(f3);
			termPtrOutputStream = new FileOutputStream(f4);
			rIdToPIdOutputStream = new FileOutputStream(f5);
			freqOutputStream = new FileOutputStream(f6);
			invertedIndexOutputStream = new FileOutputStream(f8);
			invertedIndexPtrOutputStream = new FileOutputStream(f9);
		}

		void close() throws IOException {
			termOutputStream.close();
			prefixOutputStream.close();
			lengthOutputStream.close();
			termPtrOutputStream.close();
			freqOutputStream.close();
			invertedIndexOutputStream.close();
			invertedIndexPtrOutputStream.close();
		}

	}

	public static class InvertedCompressResult {
		public List<Byte> result;
		public List<Byte> invertedIndexPtr;

		InvertedCompressResult(List<Byte> result, List<Byte> invertedIndexPtr) {
			this.result = result;
			this.invertedIndexPtr = invertedIndexPtr;
		}
	}

	public static class CompressedDictResult {
		public List<Integer> lengths;
		public List<Integer> prefixes;
		public List<Integer> termPtr;
		public List<Integer> invertedIndexRId;
		public List<Integer> invertedIndexFreqInRid;
		public String term;


		CompressedDictResult(List<Integer> lengths,
							 List<Integer> prefixes,
							 List<Integer> termPtr,
							 List<Integer> invertedIndexRId,
							 List<Integer> invertedIndexFreqInRid,
							 String term) {
			this.lengths = lengths;
			this.prefixes = prefixes;
			this.termPtr = termPtr;
			this.invertedIndexRId = invertedIndexRId;
			this.invertedIndexFreqInRid = invertedIndexFreqInRid;
			this.term = term;
		}

		CompressedDictResult(
				List<Integer> lengths,
				List<Integer> prefixes,
				List<Integer> termPtr
		) {
			this(lengths, prefixes, termPtr, null, null, "");
		}
	}

	private class InfoWriter {

		private final GammaCode.BitWriterToFile prefixWriter;
		private final GammaCode.BitWriterToFile lengthWriter;
		private final GammaCode.BitWriterToFile termPtrOutputStream;
		private final OutputStream termOutput;
		private final GammaCode.BitWriterToFile freqOutputStream;
		private final GammaCode.BitWriterToFile invertedIndexOutputStream;
		private final GammaCode.BitWriterToFile invertedIndexPtrOutputStream;
		private int lastLastInvertedIndexPtr;
		private int lastTermPtr;
		private int lastInvertedIndexPtr;

		InfoWriter(OutputStreamWrapper outputStreamWrapper) {
			termOutput = new BufferedOutputStream(outputStreamWrapper.termOutputStream);
			invertedIndexOutputStream =
					new GammaCode.BitWriterToFile(outputStreamWrapper.invertedIndexOutputStream);
			prefixWriter = new GammaCode.BitWriterToFile(outputStreamWrapper.prefixOutputStream);
			lengthWriter = new GammaCode.BitWriterToFile(outputStreamWrapper.lengthOutputStream);
			termPtrOutputStream = new GammaCode.BitWriterToFile(outputStreamWrapper.termPtrOutputStream);
			freqOutputStream = new GammaCode.BitWriterToFile(outputStreamWrapper.freqOutputStream);
			invertedIndexPtrOutputStream = new GammaCode.BitWriterToFile(outputStreamWrapper.invertedIndexPtrOutputStream);
			this.lastTermPtr = 0;
			this.lastInvertedIndexPtr = 0;
			this.lastLastInvertedIndexPtr = 0;
		}

		void writeFinalResult() {
			prefixWriter.writeFinalResult();
			lengthWriter.writeFinalResult();
			termPtrOutputStream.writeFinalResult();
			freqOutputStream.writeFinalResult();
			invertedIndexPtrOutputStream.writeFinalResult();
			invertedIndexOutputStream.writeFinalResult();
			try {
				termOutput.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

		void writeInfoToFile(MergerKInKMinusOneFrontCoding.WriteInfo info, int freq) {
			try {
				termOutput.write(info.word.getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
			if (info.prefix != -1) {
				prefixesCoder.encodeSingleNumber(this.prefixWriter, info.prefix, 0);
			}
			if (info.length != -1) {
				lengthsCoder.encodeSingleNumber(this.lengthWriter, info.length, 0);
			}
			if (info.termPtr != -1) {
				termPtrCoder.encodeSingleNumber(this.termPtrOutputStream, info.termPtr, lastTermPtr);
				lastTermPtr = info.termPtr;
			}
			tokenFrequencyCoder.encodeSingleNumber(this.freqOutputStream, freq, 0);
		}

		public void writeInvertedIndexes(List<List<Integer>> inverteds) {
			int lastInvertedValue = 0;
			for (List<Integer> inverted : inverteds) {
				for (int i = 0; i < inverted.size(); i += 1) {
					int value = inverted.get(i);
					int valueToSave = value - lastInvertedValue;
					invertedIndexCoder.encodeSingleNumber(this.invertedIndexOutputStream, valueToSave, 0);
					lastInvertedValue = value;
				}
			}
			this.invertedIndexOutputStream.writeFinalResult(false);
			invertedIndexPtrCoder.encodeSingleNumber(this.invertedIndexPtrOutputStream, this.lastInvertedIndexPtr, this.lastLastInvertedIndexPtr);
			this.lastLastInvertedIndexPtr = this.lastInvertedIndexPtr;
			this.lastInvertedIndexPtr = this.invertedIndexOutputStream.getCurrentByteCount();
		}
	}

	public class WordInfoIter implements Iterator<WordInfo> {

		private final InputStream termInputStream;
		private final GammaCode.BitReaderFromFile termPtrReader;
		private final GammaCode.BitReaderFromFile prefixReader;
		private final GammaCode.BitReaderFromFile lengthReader;
		private final GammaCode.BitReaderFromFile freqReader;
		private final int blockSize;
		private final GammaCode.BitReaderFromFile invertedIndexPtrReader;
		private final InputStream invertedIndexInputStream;
		private int lastInvertedPtr;
		private int lastTermPtr;
		private String wordBlock;
		private int currentTermPtr;
		private int nextWordIndex;
		private WordInfo nextInfo;

		public WordInfoIter(InputStreamWrapper inputStreamWrapper) {
			this.termInputStream = new BufferedInputStream(inputStreamWrapper.termInputStream);
			this.invertedIndexInputStream =
					new BufferedInputStream(inputStreamWrapper.invertedIndexInputStream);
			prefixReader = new GammaCode.BitReaderFromFile(inputStreamWrapper.prefixInputStream);
			lengthReader = new GammaCode.BitReaderFromFile(inputStreamWrapper.lengthInputStream);
			termPtrReader = new GammaCode.BitReaderFromFile(inputStreamWrapper.termPtrInputStream);
			freqReader = new GammaCode.BitReaderFromFile(inputStreamWrapper.freqInputStream);

			invertedIndexPtrReader =
					new GammaCode.BitReaderFromFile(inputStreamWrapper.invertedIndexPtrInputStream);

			termPtrCoder.getValue(termPtrReader, 0); // Dropping the 0.
			invertedIndexPtrCoder.getValue(invertedIndexPtrReader, 0); // Dropping the 0.
			this.lastTermPtr = 0;
			this.lastInvertedPtr = 0;
			this.wordBlock = null;
			this.nextWordIndex = 0;
			this.blockSize = dictCoder.block_size;
			this.currentTermPtr = 0;
			this.nextInfo = this.readNextWord();
		}

		public void terminate() {
			try {
				this.termInputStream.close();
				this.invertedIndexInputStream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		private WordInfo readNextWord() {
			int prefix;
			if (nextWordIndex % blockSize == 0) {
				prefix = 0;
			} else {
				prefix = prefixesCoder.getValue(prefixReader, 0);
			}
			int length;
			if (nextWordIndex % blockSize == blockSize - 1) {
				int value = termPtrCoder.getValue(termPtrReader, lastTermPtr);
				lastTermPtr = value;
				length = value - currentTermPtr + prefix;
				if (length < 0) {
					length = 100; // TODO check this.
				}
			} else {
				length = lengthsCoder.getValue(lengthReader, 0);
			}
			byte[] buf = new byte[length - prefix];
			try {
				termInputStream.read(buf);
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException("readNextInfo");
			}
			String subWord = new String(buf);
			if (nextWordIndex % blockSize == 0) {
				this.wordBlock = subWord;
			}
			String word = wordBlock.substring(0, prefix).concat(subWord);
			word = word.replaceAll("\u0000", ""); // if we read too much (end of file), then it's removed.
			if (word.length() == 0) {
				return null;
			}
			currentTermPtr += (length - prefix);
			nextWordIndex++;
			int freq = tokenFrequencyCoder.getValue(this.freqReader, 0);

			// Inverted
			int nextInverted = invertedIndexPtrCoder.getValue(invertedIndexPtrReader, this.lastInvertedPtr);
			if (nextInverted <= this.lastInvertedPtr) {
				buf = new byte[100];
			} else {
				buf = new byte[nextInverted - this.lastInvertedPtr];
			}
			this.lastInvertedPtr = nextInverted;
			try {
				this.invertedIndexInputStream.read(buf);
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException("readNextInfo");
			}
			List<Integer> invertedIndex = decompressInvertedIndex(buf, freq);
			return new WordInfo(word, freq, 0, invertedIndex);
		}

		@Override
		public boolean hasNext() {
			return this.nextInfo != null;
		}

		@Override
		public WordInfo next() {
			WordInfo temp = this.nextInfo;
			this.nextInfo = this.readNextWord();
			return temp;
		}

	}

}
