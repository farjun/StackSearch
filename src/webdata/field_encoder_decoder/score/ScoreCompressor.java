package webdata.field_encoder_decoder.score;

import webdata.compression.GammaCode;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

public class ScoreCompressor {

	private final GammaCode coder;

	public ScoreCompressor() {
		this.coder = new GammaCode();
	}


	public List<Byte> compress(List<Integer> numbers) {
		return this.coder.encode(numbers);
	}

	public List<Integer> decompress(byte[] bytes, int size) {
		return this.coder.decode(size, bytes);
	}


	public void mergeInputsToOutput(
			int amountToRead1,
			int amountToRead2,
			InputStream inputStream1,
			InputStream inputStream2,
			OutputStream outputStream
	) {
		this.coder.mergeTwoInputToOutput(amountToRead1, amountToRead2, inputStream1, inputStream2, outputStream);
	}

	public void megaMergeInputsToOutput(
			List<Integer> amountsToRead,
			List<InputStream> inputStreams,
			OutputStream outputStream
	) {
		this.coder.mergeInputsToOutput(amountsToRead, inputStreams, outputStream);
	}
}
