package webdata.compression;

public class DeltaCodeGap extends DeltaCode {

	public DeltaCodeGap() {
		super();
	}

	public DeltaCodeGap(int addForEachNumber) {
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
