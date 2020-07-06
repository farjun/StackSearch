package webdata.compression;

import java.util.List;

public class GammaCodeGap extends GammaCode {


	public GammaCodeGap() {
		super();
	}

	public GammaCodeGap(int addForEachNumber) {
		super(addForEachNumber);
	}

	@Override
	public void encodeSingleNumber(GammaCode.BitWriter bitWriter, Integer number, Integer lastValue) {
		super.encodeSingleNumber(bitWriter, number - lastValue, lastValue);
	}

	@Override
	public int getValue(GammaCode.BitReaderBase bitReader, Integer lastValue) {
		return super.getValue(bitReader, lastValue) + lastValue;
	}

}
