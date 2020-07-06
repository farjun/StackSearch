package webdata;


import webdata.field_encoder_decoder.helpfulness.HelpfulnessEncoder;
import webdata.field_encoder_decoder.meta.MetaEncoder;
import webdata.field_encoder_decoder.productId.ProductIdEncoder;
import webdata.field_encoder_decoder.score.ScoreEncoder;
import webdata.field_encoder_decoder.text.TextEncoder;
import webdata.review.Review;
import webdata.review.ReviewParser;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class SlowIndexWriter {

	/**
	 * Given product review data, creates an on disk index
	 * inputFile is the path to the file
	 * containing the review data
	 * dir is the directory in which all index files will be created
	 * if the directory does not exist, it should be created
	 */
	public void slowWrite(String inputFile, String dir) {
		List<Review> reviews = getReviews(inputFile);
		writeReviewsToDir(dir, reviews);
	}

	private List<Review> getReviews(String inputFile) {
		FileInputStream inputStream = null;
		Scanner scanner = null;
		List<Review> reviews = new ArrayList<>();
		try {
			inputStream = new FileInputStream(inputFile);
			scanner = new Scanner(inputStream);
			while (scanner.hasNextLine()) {
				Review review = getReview(scanner);
				reviews.add(review);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			doClose(inputStream, scanner);
		}
		return reviews;
	}

	private void writeReviewsToDir(String dir, List<Review> reviews) {
		initMeta(dir);
		MetaEncoder.getInstance().setReviewSize(reviews.size());
		write_reviews_productId(dir, reviews);
		write_reviews_text(dir, reviews);
		write_reviews_score(dir, reviews);
		write_reviews_helpfulness(dir, reviews);
		writeMeta();
	}

	private void writeMeta() {
		MetaEncoder.getInstance().encode();
	}

	private void initMeta(String dir) {
		MetaEncoder.getInstance().setDir(dir);
	}

	private void write_reviews_productId(String dir, List<Review> reviews) {
		List<String> productIds =
				reviews.stream().map((Review r) -> r.productId).collect(Collectors.toList());
		new ProductIdEncoder(dir).encode(productIds);
	}

	private void write_reviews_helpfulness(String dir, List<Review> reviews) {
		List<String> helpfulness =
				reviews.stream().map((Review r) -> r.helpfulness).collect(Collectors.toList());
		new HelpfulnessEncoder(dir).encode(helpfulness);
	}

	private void write_reviews_score(String dir, List<Review> reviews) {
		List<String> scores =
				reviews.stream().map((Review r) -> r.score.substring(0, 1)).collect(Collectors.toList());
		new ScoreEncoder(dir).encode(scores);
	}

	private void write_reviews_text(String dir, List<Review> reviews) {
		List<String> texts =
				reviews.stream().map((Review r) -> r.text).collect(Collectors.toList());
		new TextEncoder(dir).encode(texts);
	}

	private void doClose(FileInputStream inputStream, Scanner scanner) {
		if (inputStream != null) {
			try {
				inputStream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (scanner != null) {
			scanner.close();
		}
	}

	private Review getReview(Scanner scanner) {
		List<String> list = new ArrayList<>();
		String data = scanner.nextLine();
		while (!data.equals("")) {
			list.add(data);
			if (scanner.hasNextLine()) {
				data = scanner.nextLine();
			} else {
				data = "";
			}
		}
		return ReviewParser.parseReview(list);
	}

	/**
	 * Delete all index files by removing the given directory
	 */
	public void removeIndex(String dir) {
		File dirAsFile = new File(dir);
		if (!dirAsFile.exists()) {
			return;
		}
		File[] filesInDir = dirAsFile.listFiles();
		for (File file : filesInDir) {
			file.delete();
		}
		dirAsFile.delete();
	}
}