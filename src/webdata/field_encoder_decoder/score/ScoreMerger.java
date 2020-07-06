package webdata.field_encoder_decoder.score;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ScoreMerger {

	public static void merge(
			String inDir1,
			int amountToRead1,
			String inDir2,
			int amountToRead2,
			String outDir) {
		String fName1 = Paths.get(inDir1, ScoreConstants.OUT_FILE_NAME.value).toString();
		String fName2 = Paths.get(inDir2, ScoreConstants.OUT_FILE_NAME.value).toString();
		String fOut = Paths.get(outDir, ScoreConstants.OUT_FILE_NAME.value).toString();
		try {
			OutputStream outputStream = new FileOutputStream(fOut);
			InputStream inputStream1 = new FileInputStream(fName1);
			InputStream inputStream2 = new FileInputStream(fName2);
			new ScoreCompressor().mergeInputsToOutput(
					amountToRead1, amountToRead2, inputStream1, inputStream2, outputStream
			);
			inputStream1.close();
			inputStream2.close();
			outputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static void megaMerge(List<Integer> amountOfReviews, List<String> inDirs, String out) {
		try {
			List<InputStream> inputStreams = new ArrayList<>();
			for (String inDir : inDirs) {
				inputStreams.add(new FileInputStream(Paths.get(inDir, ScoreConstants.OUT_FILE_NAME.value).toString()));
			}
			OutputStream outputStream = new FileOutputStream(Paths.get(out, ScoreConstants.OUT_FILE_NAME.value).toString());
			new ScoreCompressor().megaMergeInputsToOutput(amountOfReviews, inputStreams, outputStream);
			for (InputStream inputStream : inputStreams) {
				inputStream.close();
			}
			outputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
