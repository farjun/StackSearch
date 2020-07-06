package webdata;


import webdata.field_encoder_decoder.Encoder;
import webdata.field_encoder_decoder.helpfulness.HelpfulnessEncoder;
import webdata.field_encoder_decoder.helpfulness.HelpfulnessMerger;
import webdata.field_encoder_decoder.meta.MetaEncoder;
import webdata.field_encoder_decoder.productId.ProductIdEncoder;
import webdata.field_encoder_decoder.productId.ProductIdMerger;
import webdata.field_encoder_decoder.score.ScoreEncoder;
import webdata.field_encoder_decoder.score.ScoreMerger;
import webdata.field_encoder_decoder.text.TextEncoder;
import webdata.field_encoder_decoder.text.TextMerger;
import webdata.review.Review;
import webdata.review.ReviewIterator;
import webdata.utils.Utils;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class IndexWriter {

	private Encoder helpfulnessEncoder;
	private Encoder scoreEncoder;
	private Encoder productIdEncoder;
	private Encoder textEncoder;
	private int stopAfter;
	private Encoder[] encoders;

	public IndexWriter(int stopAfter) {
		this.stopAfter = stopAfter;
	}

	public IndexWriter() {
		this(-1);
	}


	/**
	 * Given product review data, creates an on disk index
	 * inputFile is the path to the file
	 * containing the review data
	 * dir is the directory in which all index files will be created
	 * if the directory does not exist, it should be created
	 */
	public void write(String inputFile, String dir) {
		List<Integer> reviewInBatch = writeMidResults(inputFile, dir);
		this.megaMergeMidResults(dir, reviewInBatch);
	}

	private List<Integer> writeMidResults(String inputFile, String dir) {
		int batchFolderIndex = 0;
		this.initEncoders(this.getDirName(dir, batchFolderIndex));
		ReviewIterator reviewIterator = new ReviewIterator(inputFile);
		Runtime runtime = Runtime.getRuntime();
		List<Integer> reviewInBatch = new ArrayList<>();
		Review review = null;
		boolean haveWrittenNewReview = false;
		long threshold = runtime.maxMemory() / 2; // TODO try to think on other safety
		while (reviewIterator.hasNext()) {
			long allocatedMemory = (runtime.totalMemory() - runtime.freeMemory());
			long freeMemory = runtime.maxMemory() - allocatedMemory;
			if (freeMemory < threshold && haveWrittenNewReview) {
				haveWrittenNewReview = false;
				batchFolderIndex = writeMidResult(dir, batchFolderIndex, reviewInBatch, review);
			} else {
				// Else add review to encoders.
				haveWrittenNewReview = true;
				review = reviewIterator.next();
				addReviewToEncoders(review);
				if (this.stopAfter != -1 && review.id == this.stopAfter - 1) {
					break;
				}
			}
		}
		if (haveWrittenNewReview) {
			batchFolderIndex = writeMidResult(dir, batchFolderIndex, reviewInBatch, review);
		}
		reviewIterator.killMe();
		return reviewInBatch;
	}

	private int writeMidResult(String dir, int batchFolderIndex, List<Integer> reviewInBatch, Review review) {
		int prevId = reviewInBatch.stream().reduce(-1, Integer::sum);
		reviewInBatch.add(review.id - prevId);
//		System.out.printf("batchFolderIndex: %d, review in batch: %d, total: %d\n", batchFolderIndex,
//				review.id - prevId, review.id + 1);
		batchFolderIndex = wrapMidResult(dir, batchFolderIndex);
		return batchFolderIndex;
	}

	private void megaMergeMidResults(String dir, List<Integer> reviewInBatch) {
		int start = 0;
		int end = reviewInBatch.size();
		if (end == 1) {
			Utils.copyFolder(Paths.get(dir, "0").toFile(), new File(dir));
			this.removeIndex(Paths.get(dir, "0").toString());
			return;
		}
		int batchSize = 50;
		List<String> inDirs;
		List<Integer> reviews;
		List<Integer> leftOvers = new ArrayList<>();
		while (end - start + leftOvers.size() > batchSize) {
			int i = start;
			while (i < end) {
				if (i < end - batchSize) {
					inDirs = new ArrayList<>();
					reviews = new ArrayList<>();
					int outSize = 0;
					for (int j = i; j < i + batchSize; j++) {
						inDirs.add(this.getDirName(dir, j));
						Integer size = reviewInBatch.get(j);
						reviews.add(size);
						outSize += size;
					}
					String outDir = this.getDirName(dir, reviewInBatch.size());
					this.megaMergeDirs(inDirs, reviews, outDir);
					reviewInBatch.add(outSize);
					i += batchSize;
				} else if (i - leftOvers.size() < end - batchSize) {
					inDirs = new ArrayList<>();
					reviews = new ArrayList<>();
					int outSize = 0;
					for (int j = i; j < end; j++) {
						inDirs.add(this.getDirName(dir, j));
						Integer size = reviewInBatch.get(j);
						reviews.add(size);
						outSize += size;
					}
					int taken = inDirs.size();
					for (int j = 0; j < batchSize - taken; j++) {
						int leftover = leftOvers.remove(0);
						inDirs.add(this.getDirName(dir, leftover));
						Integer size = reviewInBatch.get(leftover);
						reviews.add(size);
						outSize += size;
					}
					String outDir = this.getDirName(dir, reviewInBatch.size());
					this.megaMergeDirs(inDirs, reviews, outDir);
					reviewInBatch.add(outSize);
					i += taken;
				} else {
					break;
				}
			}
			for (; i < end; i++) {
				leftOvers.add(i);
			}
			start = i;
			end = reviewInBatch.size();
		}
		inDirs = new ArrayList<>();
		reviews = new ArrayList<>();
		int outSize = 0;
		for (int i = start; i < end; i++) {
			inDirs.add(this.getDirName(dir, i));
			Integer size = reviewInBatch.get(i);
			reviews.add(size);
			outSize += size;
		}
		int leftOverSize = leftOvers.size();
		for (int j = 0; j < leftOverSize; j++) {
			Integer leftover = leftOvers.remove(0);
			inDirs.add(this.getDirName(dir, leftover));
			Integer size = reviewInBatch.get(leftover);
			reviews.add(size);
			outSize += size;
		}
		this.megaMergeDirs(inDirs, reviews, dir, true);
	}

	private void mergeDirs(String dir1, int amountOfReviews1, String dir2, int amountOfReviews2, String out) {
		this.mergeDirs(dir1, amountOfReviews1, dir2, amountOfReviews2, out, false);
	}

	private void megaMergeDirs(List<String> inDirs, List<Integer> amountOfReviews, String out) {
		this.megaMergeDirs(inDirs, amountOfReviews, out, false);
	}

	private void megaMergeDirs(List<String> inDirs, List<Integer> amountOfReviews, String out, boolean writeToMeta) {
		new File(out).mkdirs();
//		System.out.printf("Write mid size of: %d%n", amountOfReviews.stream().reduce(0, Integer::sum));
		ScoreMerger.megaMerge(amountOfReviews, inDirs, out);
		HelpfulnessMerger.megaMerge(amountOfReviews, inDirs, out);
		int amountOfTokens = TextMerger.megaMerge(amountOfReviews, inDirs, out);
		int amountOfPIds = ProductIdMerger.megaMerge(amountOfReviews, inDirs, out);
		for (String inDir : inDirs) {
			this.removeIndex(inDir);
		}
		if (writeToMeta) {
			MetaEncoder instance = MetaEncoder.getInstance();
			instance.setDir(out);
			instance.setReviewSize(amountOfReviews.stream().reduce(0, Integer::sum));
			instance.setTokenSize(amountOfTokens);
			instance.setPIdSize(amountOfPIds);
			instance.encode();
		}
	}

	private void mergeDirs(String dir1, int amountOfReviews1, String dir2, int amountOfReviews2, String out
			, boolean writeToMeta) {
//		System.out.printf("Write mid size of: %d%n", amountOfReviews1 + amountOfReviews2);
		new File(out).mkdirs();
		ScoreMerger.merge(dir1, amountOfReviews1, dir2, amountOfReviews2, out);
		HelpfulnessMerger.merge(dir1, amountOfReviews1, dir2, amountOfReviews2, out);
		int amountOfTokens = TextMerger.merge(dir1, amountOfReviews1, dir2, amountOfReviews2, out);
		int amountOfPIds = ProductIdMerger.merge(dir1, amountOfReviews1, dir2, amountOfReviews2, out);
		this.removeIndex(dir1);
		this.removeIndex(dir2);
		if (writeToMeta) {
			MetaEncoder instance = MetaEncoder.getInstance();
			instance.setDir(out);
			instance.setReviewSize(amountOfReviews1 + amountOfReviews2);
			instance.setTokenSize(amountOfTokens);
			instance.setPIdSize(amountOfPIds);
			instance.encode();
		}

	}

	private void initEncoders(String dir) {
		helpfulnessEncoder = new HelpfulnessEncoder(dir);
		scoreEncoder = new ScoreEncoder(dir);
		productIdEncoder = new ProductIdEncoder(dir);
		textEncoder = new TextEncoder(dir);
		MetaEncoder.getInstance().setDir(dir);
		encoders = new Encoder[]{
				helpfulnessEncoder,
				scoreEncoder,
				productIdEncoder,
				textEncoder
		};

	}

	private int wrapMidResult(String dir, int batchFolderIndex) {
		int batchIndex = batchFolderIndex + 1;
		String newDir = getDirName(dir, batchIndex);
		writeMidResult(newDir);
		return batchIndex;
	}

	private String getDirName(String dir, int batchIndex) {
		return Paths.get(dir, String.valueOf(batchIndex)).toString();
	}

	private void addReviewToEncoders(Review review) {
		this.helpfulnessEncoder.add(review.id, review.helpfulness);
		this.scoreEncoder.add(review.id, review.score);
		this.textEncoder.add(review.id, review.text);
		this.productIdEncoder.add(review.id, review.productId);
	}

	private void writeMidResult(String newDir) {
		for (Encoder encoder : encoders) {
			encoder.saveResult();
			encoder.reset(newDir);
		}
		MetaEncoder.getInstance().saveResult();
		System.gc();
	}

	/**
	 * Delete all index files by removing the given directory
	 */
	public void removeIndex(String dir) {
		File dirAsFile = new File(dir);
		if (!dirAsFile.exists()) {
			return;
		}
		this.removeIndexRec(dirAsFile);
	}

	public void removeIndexRec(File dirAsFile) {
		File[] filesInDir = dirAsFile.listFiles();
		for (File file : filesInDir) {
			if (file.isDirectory()) {
				this.removeIndexRec(file);
			}
			file.delete();
		}
		dirAsFile.delete();
	}


}