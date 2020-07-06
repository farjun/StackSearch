package webdata.review;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

public class ReviewIterator implements Iterator<Review> {

	private final InputStream inputStream;
	private final InputStream bufferInputStream;
	private final Scanner scanner;

	public ReviewIterator(String inputFile) {
		ReviewParser.reset();
		try {
			FileInputStream fInputStream = new FileInputStream(inputFile);
			if (inputFile.endsWith(".gz")) {
				inputStream = new GZIPInputStream(fInputStream);
			} else {
				inputStream = fInputStream;
			}
			bufferInputStream = new BufferedInputStream(inputStream);
			scanner = new Scanner(bufferInputStream);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new RuntimeException("ReviewIterator throws exception");
		} catch (IOException e) {
			throw new RuntimeException("ReviewIterator throws exception");
		}
	}

	public void killMe() {
		try {
			bufferInputStream.close();
			inputStream.close();
			scanner.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public boolean hasNext() {
		return scanner.hasNextLine();
	}

	@Override
	public Review next() {
		return getReview(scanner);
	}

	private Review getReview(Scanner scanner) {
		List<String> list = new ArrayList<>();
		boolean terminate = false;
		boolean productId = false;
		boolean helpfulness = false;
		boolean score = false;
		boolean text = false;
		String data;
		while (!terminate) {
			if (scanner.hasNextLine()) {
				data = scanner.nextLine();
				productId = productId || data.startsWith("product/productId:");
				helpfulness = helpfulness || data.startsWith("review/helpfulness");
				score = score || data.startsWith("review/score:");
				text = text || data.startsWith("review/text:");
				terminate = text && score && helpfulness && productId && data.equals("");
				list.add(data);
			} else {
				terminate = true;
			}
		}
		return ReviewParser.parseReview(list);
	}


}
