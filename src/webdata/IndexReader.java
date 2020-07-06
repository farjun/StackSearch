package webdata;

import webdata.field_encoder_decoder.helpfulness.HelpfulnessDecoder;
import webdata.field_encoder_decoder.meta.MetaDecoder;
import webdata.field_encoder_decoder.productId.ProductIdDecoder;
import webdata.field_encoder_decoder.score.ScoreDecoder;
import webdata.field_encoder_decoder.text.TextDecoder;

import java.util.*;

public class IndexReader {

	private final ScoreDecoder scoreDecoder;
	private final HelpfulnessDecoder helpfulnessDecoder;
	private final int numReviews;
	private final ProductIdDecoder productIdDecoder;
	private final TextDecoder textDecoder;
	private final int tokenSizeOfReviews;

	/**
	 * Creates an IndexReader which will read from the given directory
	 */
	public IndexReader(String dir) {
		MetaDecoder metaDecoder = new MetaDecoder(dir);
		numReviews = metaDecoder.reviewSize;
		int numTokens = metaDecoder.tokenSize;
		int pIdSize = metaDecoder.pIdSize;
		this.scoreDecoder = new ScoreDecoder(dir, numReviews);
		this.helpfulnessDecoder = new HelpfulnessDecoder(dir, numReviews);
		this.productIdDecoder = new ProductIdDecoder(dir, numReviews, pIdSize);
		this.textDecoder = new TextDecoder(dir, numReviews, numTokens);
		this.tokenSizeOfReviews = this.textDecoder.getTokenSizeOfReviews();

	}

	/**
	 * Returns the product identifier for the given review
	 * Returns null if there is no review with the given identifier
	 */
	public String getProductId(int reviewId) {
		if (this.is_review_id_valid(reviewId)) {
			return this.productIdDecoder.getProductIdByReviewId(reviewId - 1);
		}
		return null;
	}

	/**
	 * Returns the score for a given review
	 * Returns -1 if there is no review with the given identifier
	 */
	public int getReviewScore(int reviewId) {
		if (this.is_review_id_valid(reviewId)) {
			return scoreDecoder.decodeByReviewID(reviewId - 1);
		}
		return -1;
	}

	/**
	 * Returns the numerator for the helpfulness of a given review
	 * Returns -1 if there is no review with the given identifier
	 */
	public int getReviewHelpfulnessNumerator(int reviewId) {
		if (this.is_review_id_valid(reviewId)) {
			return this.helpfulnessDecoder.getNumeratorByReviewID(reviewId - 1);
		}
		return -1;
	}

	/**
	 * Returns the denominator for the helpfulness of a given review
	 * Returns -1 if there is no review with the given identifier
	 */
	public int getReviewHelpfulnessDenominator(int reviewId) {
		if (this.is_review_id_valid(reviewId)) {
			return this.helpfulnessDecoder.getDenominatorByReviewID(reviewId - 1);
		} else {
			return -1;
		}
	}

	/**
	 * Returns the number of tokens in a given review
	 * Returns -1 if there is no review with the given identifier
	 */
	public int getReviewLength(int reviewId) {
		if (this.is_review_id_valid(reviewId)) {
			return this.textDecoder.getReviewLengthByReviewID(reviewId - 1);
		}
		return -1;
	}

	/**
	 * Return the number of reviews containing a given token (i.e., word)
	 * Returns 0 if there are no reviews containing this token
	 */
	public int getTokenFrequency(String token) {
		token = token.toLowerCase();
		return this.textDecoder.getTokenFrequencyByToken(token);
	}

	/**
	 * Return the number of times that a given token (i.e., word) appears in
	 * the reviews indexed
	 * Returns 0 if there are no reviews containing this token
	 */
	public int getTokenCollectionFrequency(String token) {
		token = token.toLowerCase();
		return this.textDecoder.getTokenCollectionFrequencyByToken(token);
	}

	/**
	 * Return a series of integers of the form id-1, freq-1, id-2, freq-2, ... such
	 * that id-n is the n-th review containing the given token and freq-n is the
	 * number of times that the token appears in review id-n
	 * Note that the integers should be sorted by id
	 * <p>
	 * Returns an empty Enumeration if there are no reviews containing this token
	 */
	public Enumeration<Integer> getReviewsWithToken(String token) {
		token = token.toLowerCase();
		List<Integer> invertedIndexList = this.textDecoder.getInvertedIndexList(token);
		return new ListEnumeration(invertedIndexList);
	}

	/**
	 * Return the number of product reviews available in the system
	 */
	public int getNumberOfReviews() {
		return this.numReviews;
	}

	/**
	 * Return the number of number of tokens in the system
	 * (Tokens should be counted as many times as they appear)
	 */
	public int getTokenSizeOfReviews() {
		return this.tokenSizeOfReviews;
	}

	/**
	 * Return the ids of the reviews for a given product identifier
	 * Note that the integers returned should be sorted by id
	 * <p>
	 * Returns an empty Enumeration if there are no reviews for this product
	 */
	public Enumeration<Integer> getProductReviews(String productId) {
		List<Integer> invertedIndexList = this.productIdDecoder.getInvertedIndexList(productId);
		return new ListEnumeration(invertedIndexList);
	}

	private boolean is_review_id_valid(int review_id) {
		return 0 < review_id && review_id <= this.getNumberOfReviews();
	}

	private static class ListEnumeration implements Enumeration<Integer> {

		private final Iterator<Integer> iterator;

		public ListEnumeration(List<Integer> list) {
			if (list != null) {
				iterator = list.iterator();
			} else {
				iterator = Collections.emptyIterator();
			}
		}

		/**
		 * Tests if this enumeration contains more elements.
		 *
		 * @return <code>true</code> if and only if this enumeration object
		 * contains at least one more element to provide;
		 * <code>false</code> otherwise.
		 */
		@Override
		public boolean hasMoreElements() {
			return this.iterator.hasNext();
		}

		/**
		 * Returns the next element of this enumeration if this enumeration
		 * object has at least one more element to provide.
		 *
		 * @return the next element of this enumeration.
		 * @throws NoSuchElementException if no more elements exist.
		 */
		@Override
		public Integer nextElement() {
			return iterator.next();
		}
	}
}