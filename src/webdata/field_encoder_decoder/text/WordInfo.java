package webdata.field_encoder_decoder.text;

import java.util.List;

public class WordInfo {
	public final String word;
	public final int freq;
	public final int collectionFreq;
	public final List<Integer> invertedIndex;

	public WordInfo(String word, int freq, int collectionFreq, List<Integer> invertedIndex) {
		this.word = word;
		this.freq = freq;
		this.collectionFreq = collectionFreq;
		this.invertedIndex = invertedIndex;
	}
}
