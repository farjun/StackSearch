package webdata.field_encoder_decoder.helpfulness;

import webdata.field_encoder_decoder.Encoder;
import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class HelpfulnessEncoder implements Encoder {

	private String dir;
	private List<Integer> helpfulnessDenominator;
	private List<Integer> helpfulnessNumerator;

	public HelpfulnessEncoder(String dir) {
		this.reset(dir);
	}

	public void reset(String dir) {
		this.dir = dir;
		helpfulnessDenominator = new ArrayList<>();
		helpfulnessNumerator = new ArrayList<>();
	}

	public void encode(List<String> helpfulness) {
		for (int i = 0; i < helpfulness.size(); i++) {
			String h = helpfulness.get(i);
			this.add(i, h);
		}
		saveResult();
	}

	public void saveResult() {
		List<Byte> compressD = new HelpfulnessCompressor().compress(helpfulnessDenominator);
		List<Byte> compressN = new HelpfulnessCompressor().compress(helpfulnessNumerator);
		String dPath = Paths.get(dir, HelpfulnessConstants.HELPFULNESS_DENOMINATOR_FILE_NAME.value).toString();
		Utils.writeFile(dPath, compressD);
		String nPath = Paths.get(dir, HelpfulnessConstants.HELPFULNESS_NUMERATOR_FILE_NAME.value).toString();
		Utils.writeFile(nPath, compressN);
	}

	public void add(int id, String helpfulness) {
		String[] split = helpfulness.split("/");
		int numerator = Integer.parseInt(split[0]);
		int denominator = Integer.parseInt(split[1]);
		if (numerator > denominator) {
			denominator = numerator;
		}
		helpfulnessNumerator.add(numerator);
		helpfulnessDenominator.add(denominator - numerator); // Saving the diff
	}


}
