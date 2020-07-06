package webdata.field_encoder_decoder.helpfulness;


import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.List;

public class HelpfulnessDecoder {

	private final String dir;
	private final List<Integer> denominators;
	private final List<Integer> numerators;
	private int num_review;

	public HelpfulnessDecoder(String dir, int num_review) {
		this.dir = dir;
		this.num_review = num_review;
		this.numerators = this.encodeNumerator();
		this.denominators = this.encodeDenominator();
	}

	public int getDenominatorByReviewID(int review_id) {
		return this.denominators.get(review_id) + this.getNumeratorByReviewID(review_id);
	}

	public int getNumeratorByReviewID(int review_id) {
		return this.numerators.get(review_id);
	}

	public List<Integer> encodeDenominator() {
		String f_name = HelpfulnessConstants.HELPFULNESS_DENOMINATOR_FILE_NAME.value;
		return getInts(f_name);
	}

	public List<Integer> encodeNumerator() {
		String f_name = HelpfulnessConstants.HELPFULNESS_NUMERATOR_FILE_NAME.value;
		return getInts(f_name);
	}

	private List<Integer> getInts(String f_name) {
		byte[] data = Utils.readFile(Paths.get(dir, f_name).toString());
		return new HelpfulnessCompressor().decompress(data, num_review);
	}

}
