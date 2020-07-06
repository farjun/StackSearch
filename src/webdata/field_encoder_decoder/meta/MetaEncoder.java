package webdata.field_encoder_decoder.meta;

import webdata.field_encoder_decoder.Encoder;
import webdata.utils.Utils;

import java.nio.file.Paths;


public class MetaEncoder implements Encoder {
	private static MetaEncoder metaEncoder;
	private String dir;
	private int reviewSize = 0;
	private int tokenSize = 0;
	private int pIdSize = 0;

	public static MetaEncoder getInstance() {
		if (metaEncoder == null) {
			metaEncoder = new MetaEncoder();
		}
		return metaEncoder;
	}

	public void setDir(String dir) {
		this.dir = dir;
	}

	public void setReviewSize(int size) {
		this.reviewSize = size;
	}

	public void setTokenSize(int size) {
		this.tokenSize = size;
	}

	public void setPIdSize(int size) {
		this.pIdSize = size;
	}

	public void encode() {
		byte[] reviewSizeBytes = Utils.intToByteArray(reviewSize);
		byte[] tokenSizeBytes = Utils.intToByteArray(tokenSize);
		byte[] pIdBytes = Utils.intToByteArray(pIdSize);
		byte[] data = new byte[reviewSizeBytes.length + tokenSizeBytes.length + pIdBytes.length];
		System.arraycopy(reviewSizeBytes, 0, data, 0, 4);
		System.arraycopy(tokenSizeBytes, 0, data, 4, 4);
		System.arraycopy(pIdBytes, 0, data, 8, 4);
		Utils.writeFile(Paths.get(dir, MetaConstants.F_NAME.value).toString(), data);
	}


	@Override
	public void reset(String dir) {
		this.dir = dir;
	}

	@Override
	public void saveResult() {
		this.encode();
	}

	@Override
	public void add(int id, String data) {
		throw new RuntimeException("MetaEncoder.add");
	}
}
