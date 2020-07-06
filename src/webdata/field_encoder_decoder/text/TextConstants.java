package webdata.field_encoder_decoder.text;

public enum TextConstants {
	WORD_LEN_FILE_NAME("text_word_id_to_word_length"),
	REVIEW_LEN_FILE_NAME("text_review_id_to_review_length"),
	FREQ_FILE_NAME("text_word_id_to_freq"),
	COLLECTION_FREQ_FILE_NAME("text_word_id_to_collection_freq"),
	PREFIX_FILE_NAME("text_word_id_to_prefix"),
	TERM_PTR_FILE_NAME("text_word_id_to_term_ptr"),
	TERM_FILE_NAME("text_term"),
	INVERTED_INDEX_FILE_NAME("text_inverted_index"),
	INVERTED_INDEX_PTR_FILE_NAME("text_inverted_index_ptr");

	public final String value;

	TextConstants(String value) {
		this.value = value;
	}

}
