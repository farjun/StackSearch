package webdata.review;

public class Review {

	public final int id;
	public final String productId;
	public final String helpfulness;
	public final String score;
	public final String text;

	Review(
			int id,
			String productId,
			String helpfulness,
			String score,
			String text
	) {
		this.id = id;
		this.productId = productId;
		this.helpfulness = helpfulness;
		this.score = score;
		this.text = text;
	}

}
