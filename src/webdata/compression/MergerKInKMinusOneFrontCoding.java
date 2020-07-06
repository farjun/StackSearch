package webdata.compression;


import webdata.utils.TextUtils;

public class MergerKInKMinusOneFrontCoding {
	public final int block_size;
	private int termPtr;
	private int word_idx;
	private String block_word;

	public MergerKInKMinusOneFrontCoding(int block_size) {
		this.block_size = block_size;
		this.block_word = null;
		this.word_idx = 0;
		this.termPtr = 0;
	}

	public WriteInfo addWord(String word) {
		int length = word.length();
		WriteInfo info;
		if (this.word_idx % block_size == 0) {
			this.block_word = word;
			info = new WriteInfo(word, -1, length, this.termPtr);
		} else {
			int prefix = TextUtils.getPrefix(this.block_word, word);
			String suffix = word.substring(prefix);
			if (this.word_idx % block_size == block_size - 1) {
				info = new WriteInfo(suffix, prefix, -1, -1);
			} else {
				info = new WriteInfo(suffix, prefix, length, -1);
			}
		}
		this.word_idx++;
		this.termPtr += info.word.length();
		return info;
	}

	public static class WriteInfo {

		public final String word;
		public final int prefix;
		public final int length;
		public final int termPtr;

		WriteInfo(String word, int prefix, int length, int termPtr) {

			this.word = word;
			this.prefix = prefix;
			this.length = length;
			this.termPtr = termPtr;
		}

	}

}
