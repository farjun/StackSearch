package webdata.field_encoder_decoder.text;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TextMerger {

	public static int merge(String inDir1, int amountOfReviews1, String inDir2, int amountOfReviews2, String outDir) {
		int amountOfTokens = 0;
		try {
			TextCompressor.TextInputStreamWrapper wrapper1 = new TextCompressor.TextInputStreamWrapper(inDir1);
			TextCompressor.TextInputStreamWrapper wrapper2 = new TextCompressor.TextInputStreamWrapper(inDir2);
			TextCompressor.TextOutputStreamWrapper outputStreamWrapper = new TextCompressor.TextOutputStreamWrapper(outDir);
			TextCompressor textCompressor = new TextCompressor();
			amountOfTokens = textCompressor.mergeInputsToOutput(wrapper1, wrapper2, outputStreamWrapper);
			textCompressor.mergeFileLength(wrapper1, amountOfReviews1, wrapper2, amountOfReviews2, outputStreamWrapper);
			wrapper1.close();
			wrapper2.close();
			outputStreamWrapper.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return amountOfTokens;
	}

	public static int megaMerge(
			List<Integer> amountOfReviews,
			List<String> inDirs,
			String outDir) {
		int amountOfTokens = 0;
		try {
			List<TextCompressor.TextInputStreamWrapper> l = new ArrayList<>();
			for (String inDir : inDirs) {
				l.add(new TextCompressor.TextInputStreamWrapper(inDir));
			}
			TextCompressor.TextOutputStreamWrapper outputStreamWrapper = new TextCompressor.TextOutputStreamWrapper(outDir);
			TextCompressor textCompressor = new TextCompressor();
			amountOfTokens = textCompressor.megaMerge(l, outputStreamWrapper);
			textCompressor.megaMergeFileLength(l, amountOfReviews, outputStreamWrapper);
			for (TextCompressor.TextInputStreamWrapper wrapper : l) {
				wrapper.close();
			}
			outputStreamWrapper.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return amountOfTokens;
	}

	public static void main(String[] args) {
	}

}
