package webdata.field_encoder_decoder.score;

import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.List;

public class ScoreDecoder {

	private final String dir;
	private int num_review;
	private final List<Integer> scores;

	public ScoreDecoder(String dir, int num_review){
		this.dir = dir;
		this.num_review = num_review;
		this.scores = encodeScores();
	}
	
	public List<Integer> encodeScores(){
		String fName = Paths.get(dir, ScoreConstants.OUT_FILE_NAME.value).toString();
		byte[] data = Utils.readFile(fName);
		return new ScoreCompressor().decompress(data,num_review);
	}

	public int decodeByReviewID(int i) {
		return this.scores.get(i);
	}
}
