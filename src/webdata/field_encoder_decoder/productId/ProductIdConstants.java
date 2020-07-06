package webdata.field_encoder_decoder.productId;

public enum ProductIdConstants {

	TERM_PTR_FILE_NAME("product_id_to_term_ptr"),
	WORD_LEN_FILE_NAME("product_id_to_len"),
	PREFIX_FILE_NAME("product_id_to_prefix"),
	TERM_FILE_NAME("product_id_to_term"),
	REVIEW_ID_TO_P_ID_F_NAME("product_id_review_id_to_product_id"),
	INVERTED_INDEX_FILE_NAME("product_id_inverted_index"),
	INVERTED_INDEX_PTR_FILE_NAME("product_id_inverted_index_ptr"),
	FREQ_FILE_NAME("product_id_to_freq");

	public String value;

	ProductIdConstants(String value) {

		this.value = value;
	}
}
