package webdata.review;

import java.util.List;

public class ReviewParser {

	private static final String SEP = ": ";
	private static int incremental_id = 0;

	public static void reset(){
		incremental_id = 0;
	}

	public static Review parseReview(List<String> lines) {
		String productId = null;
		String helpfulness = null;
		String score = null;
		String text = null;
		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			productId = updateIfNeeded(line, productId, "product/productId:");
			helpfulness = updateIfNeeded(line, helpfulness, "review/helpfulness");
			score = updateIfNeeded(line, score, "review/score:");
			text = updateIfNeeded(line, text, "review/text:");
			if (text != null) {
				StringBuilder stringBuilder = new StringBuilder(text);
				i++;
				for (; i < lines.size(); i++) {
					stringBuilder
							.append(" ")
							.append(lines.get(i));
				}
				text = stringBuilder.toString();
			}
		}
		if (text != null) {
			text = text.toLowerCase();
		}
		int id = incremental_id;
		++incremental_id;
		return new Review(
				id,
				productId,
				helpfulness,
				score,
				text
		);
	}

	private static String updateIfNeeded(String line, String field, String prefix) {
		if (field == null && line.startsWith(prefix)) {
			field = ReviewParser.extractFromLine(line);
		}
		return field;
	}

	private static String extractFromLine(String line) {
		int index = line.indexOf(ReviewParser.SEP);
		return line.substring(index + ReviewParser.SEP.length());
	}

}

