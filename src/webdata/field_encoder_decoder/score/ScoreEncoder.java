package webdata.field_encoder_decoder.score;

import webdata.field_encoder_decoder.Encoder;
import webdata.field_encoder_decoder.meta.MetaEncoder;
import webdata.utils.Utils;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ScoreEncoder implements Encoder {

	private String dir;
	private List<Integer> scoresAsInt;

	public ScoreEncoder(String dir) {
		this.dir = dir;
		scoresAsInt = new ArrayList<>();
	}

	public void reset(String dir) {
		this.dir = dir;
		scoresAsInt = new ArrayList<>();
	}

	public void encode(List<String> scores) {
		for (int i = 0; i < scores.size(); i++) {
			String score = scores.get(i);
			this.add(i, score);
		}
		saveResult();
	}

	public void saveResult() {
		MetaEncoder.getInstance().setReviewSize(scoresAsInt.size());
		List<Byte> bytes = new ScoreCompressor().compress(scoresAsInt);
		Utils.writeFile(Paths.get(dir, ScoreConstants.OUT_FILE_NAME.value).toString(), bytes);
	}

	public void add(int id, String score) {
		scoresAsInt.add(Integer.parseUnsignedInt(score.substring(0, 1)));
	}


}
