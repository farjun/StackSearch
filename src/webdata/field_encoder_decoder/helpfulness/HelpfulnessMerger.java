package webdata.field_encoder_decoder.helpfulness;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class HelpfulnessMerger {

	public static void merge(
			String inDir1,
			int amountToRead1,
			String inDir2,
			int amountToRead2,
			String outDir) {
		mergeFileName(inDir1, amountToRead1, inDir2, amountToRead2, outDir, HelpfulnessConstants.HELPFULNESS_DENOMINATOR_FILE_NAME);
		mergeFileName(inDir1, amountToRead1, inDir2, amountToRead2, outDir, HelpfulnessConstants.HELPFULNESS_NUMERATOR_FILE_NAME);
	}

	private static void mergeFileName(String inDir1, int amountToRead1, String inDir2, int amountToRead2, String outDir, HelpfulnessConstants fileName) {
		String fName1 = Paths.get(inDir1, fileName.value).toString();
		String fName2 = Paths.get(inDir2, fileName.value).toString();
		File fOut = Paths.get(outDir, fileName.value).toFile();
		fOut.getParentFile().mkdirs();
		try {
			OutputStream outputStream = new FileOutputStream(fOut);
			InputStream inputStream1 = new FileInputStream(fName1);
			InputStream inputStream2 = new FileInputStream(fName2);
			new HelpfulnessCompressor().mergeInputsToOutput(
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
		megaMergeFileName(amountOfReviews, inDirs, out, HelpfulnessConstants.HELPFULNESS_DENOMINATOR_FILE_NAME);
		megaMergeFileName(amountOfReviews, inDirs, out, HelpfulnessConstants.HELPFULNESS_NUMERATOR_FILE_NAME);

	}

	private static void megaMergeFileName(List<Integer> amountOfReviews, List<String> inDirs, String out, HelpfulnessConstants fileName) {
		try {
			List<InputStream> inputStreams = new ArrayList<>();
			for (String inDir : inDirs) {
				inputStreams.add(new FileInputStream(Paths.get(inDir, fileName.value).toString()));
			}
			OutputStream outputStream = new FileOutputStream(Paths.get(out, fileName.value).toString());
			new HelpfulnessCompressor().megaMergeInputsToOutput(amountOfReviews, inputStreams, outputStream);
			for (InputStream inputStream : inputStreams) {
				inputStream.close();
			}
			outputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
