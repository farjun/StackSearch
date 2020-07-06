package webdata.field_encoder_decoder.meta;

import webdata.utils.Utils;

import java.nio.file.Paths;

public class MetaDecoder {

	public int reviewSize = 0;
	public int tokenSize = 0;
	public int pIdSize = 0;
	private String dir;

	public MetaDecoder(String dir) {
		this.dir = dir;
		this.decode();
	}

	public void decode() {
		String path = Paths.get(dir, MetaConstants.F_NAME.value).toString();
		byte[] data = Utils.readFile(path);
		this.reviewSize = Utils.byteArrayToInt(data, 0);
		this.tokenSize = Utils.byteArrayToInt(data, 4);
		this.pIdSize = Utils.byteArrayToInt(data, 8);
	}

}
