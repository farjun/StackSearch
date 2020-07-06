package webdata.compression;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class DeltaCode implements UnaryCoder {

	private final GammaCode lenEncoder;
	private int addForEachNumber;

	public DeltaCode(int addForEachNumber) {
		this.addForEachNumber = addForEachNumber;
		this.lenEncoder = new GammaCode();
	}

	public DeltaCode() {
		this(0);
	}

	public List<Byte> encode(List<Integer> numbers) {
		GammaCode.BitWriter bitWriter = new GammaCode.BitWriter();
		Integer lastValue = 0;
		for (Integer number : numbers) {
			encodeSingleNumber(bitWriter, number, lastValue);
			lastValue = number;
		}
		return bitWriter.getFinalResult();
	}

	public void encodeSingleNumber(GammaCode.BitWriter bitWriter, Integer number, Integer lastValue) {
		int size = this.lenEncoder.binarySize(number + this.addForEachNumber);
		this.lenEncoder.encodeSingleNumber(bitWriter, size, 0);
		int offsetSize = size - 1;
		int offsetAsInt = this.getOffset(offsetSize, number + this.addForEachNumber);
		bitWriter.writeBinary(offsetAsInt, offsetSize);
	}

	public int getOffset(int offsetSize, int number) {
		if (offsetSize == 0) {
			return 0;
		}
		int mask = (1 << offsetSize) - 1;
		return number & mask;
	}


	public List<Integer> decode(int amountToRead, byte[] bytes) {
		GammaCode.BitReader bitReader = new GammaCode.BitReader(bytes);
		List<Integer> result = new ArrayList<>(amountToRead);
		int lastValue = 0;
		for (int i = 0; i < amountToRead; i++) {
			int value = getValue(bitReader, lastValue);
			result.add(value);
			lastValue = value;
		}
		return result;
	}

	public int getValue(GammaCode.BitReaderBase bitReader, Integer lastValue) {
		int binaryLen = bitReader.readNextNumber();
		return bitReader.getInteger(binaryLen - 1) - this.addForEachNumber;
	}

	@Override
	public void mergeTwoInputToOutput(
			int amountToRead1, int amountToRead2, InputStream inputStream1, InputStream inputStream2, OutputStream outputStream) {
		GammaCode.BitWriterToFile writer = new GammaCode.BitWriterToFile(outputStream);
		Integer lastValue = addContentToFile(writer, inputStream1, amountToRead1, 0);
		addContentToFile(writer, inputStream2, amountToRead2, lastValue);
		writer.writeFinalResult();
	}

	public void mergeInputsToOutput(
			List<Integer> amountsToRead,
			List<InputStream> inputStream,
			OutputStream outputStream
	) {
		GammaCode.BitWriterToFile writer = new GammaCode.BitWriterToFile(outputStream);
		int lastValue = 0;
		for (int i = 0; i < amountsToRead.size(); i++) {
			lastValue = addContentToFile(writer, inputStream.get(i), amountsToRead.get(i), lastValue);
		}
		writer.writeFinalResult();
	}

	private Integer addContentToFile(GammaCode.BitWriter writer, InputStream inputStream, int amountToRead,
									 Integer lastValue) {
		GammaCode.BitReaderBase reader = new GammaCode.BitReaderFromFile(inputStream);
		for (int i = 0; i < amountToRead; i++) {
			Integer number = this.getValue(reader, lastValue);
			this.encodeSingleNumber(writer, number, lastValue);
			lastValue = number;
		}
		return lastValue;
	}

	@Override
	public void mergeTwoInputToOutputWithMap(int amountToRead1, int amountToRead2, InputStream inputStream1, InputStream inputStream2, List<Integer> map1, List<Integer> map2, OutputStream outputStream) {
		GammaCode.BitWriterToFile writer = new GammaCode.BitWriterToFile(outputStream);
		Integer lastValue = addContentToFileMap(writer, inputStream1, amountToRead1, 0, map1);
		addContentToFileMap(writer, inputStream2, amountToRead2, lastValue, map2);
		writer.writeFinalResult();
	}

	@Override
	public void mergeInputToOutputWithMaps(List<Integer> amountsToRead, List<InputStream> inputStreams, List<List<Integer>> maps, OutputStream outputStream) {
		GammaCode.BitWriterToFile writer = new GammaCode.BitWriterToFile(outputStream);
		int lastValue = 0;
		for (int i = 0; i < amountsToRead.size(); i++) {
			lastValue = addContentToFileMap(writer, inputStreams.get(i), amountsToRead.get(i), lastValue, maps.get(i));
		}
		writer.writeFinalResult();
	}

	private int addContentToFileMap(
			GammaCode.BitWriterToFile writer,
			InputStream inputStream, int amountToRead,
			int lastValue, List<Integer> map2) {
		GammaCode.BitReaderBase reader = new GammaCode.BitReaderFromFile(inputStream);
		for (int i = 0; i < amountToRead; i++) {
			int number = this.getValue(reader, lastValue);
			int MappedValue = map2.get(number);
			this.encodeSingleNumber(writer, MappedValue, lastValue);
			lastValue = number;
		}
		return lastValue;
	}
}
