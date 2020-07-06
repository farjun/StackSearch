package webdata.field_encoder_decoder.helpfulness;

public enum HelpfulnessConstants {
	HELPFULNESS_DENOMINATOR_FILE_NAME("helpfulness_denominator"),
	HELPFULNESS_NUMERATOR_FILE_NAME("helpfulness_numerator");

	public final String value;

	HelpfulnessConstants(String value){
		this.value = value;
	}

}
