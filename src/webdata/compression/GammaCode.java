package webdata.compression;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class GammaCode implements UnaryCoder {

	private int addForEachNumber;

	public GammaCode(int addForEachNumber) {
		this.addForEachNumber = addForEachNumber;
	}

	public GammaCode() {
		this(0);
	}

	public List<Byte> encode(List<Integer> numbers) {
		BitWriter bitWriter = new BitWriter();
		int lastValue = 0;
		for (Integer number : numbers) {
			encodeSingleNumber(bitWriter, number, lastValue);
			lastValue = number;
		}
		return bitWriter.getFinalResult();
	}

	public void encodeSingleNumber(BitWriter bitWriter, Integer number, Integer lastValue) {
		int binarySize = this.binarySize(number + this.addForEachNumber);
		int offsetLen = binarySize - 1;
		int offsetAsInt = this.getOffset(offsetLen, number + this.addForEachNumber);
		bitWriter.writeUnary(offsetLen);
		bitWriter.incrementCurrentIndex(); // Write Zero In the Middle.
		bitWriter.writeBinary(offsetAsInt, offsetLen);
	}

	public int binarySize(int number) {
		int value = 1;
		int binSize = 0;
		while ((value - 1) < number) {
			binSize++;
			value <<= 1;
		}
		return binSize;
	}

	public int getOffset(int offsetSize, int number) {
		if (offsetSize == 0) {
			return 0;
		}
		int mask = (1 << offsetSize) - 1;
		return number & mask;
	}

	public List<Integer> decode(int amountToRead, byte[] bytes) {
		BitReaderBase bitReader = new BitReader(bytes);
		List<Integer> result = new ArrayList<>(amountToRead);
		int lastValue = 0;
		for (int read = 0; read < amountToRead; read++) {
			int value = getValue(bitReader, lastValue);
			result.add(value);
			lastValue = value;
		}
		return result;
	}

	public int getValue(BitReaderBase bitReader, Integer lastValue) {
		return bitReader.readNextNumber() - this.addForEachNumber;
	}

	public void mergeTwoInputToOutput(
			int amountToRead1,
			int amountToRead2,
			InputStream inputStream1,
			InputStream inputStream2,
			OutputStream outputStream
	) {
		BitWriterToFile writer = new BitWriterToFile(outputStream);
		int lastValue = addContentToFile(writer, inputStream1, amountToRead1, 0);
		addContentToFile(writer, inputStream2, amountToRead2, lastValue);
		writer.writeFinalResult();
	}

	public void mergeInputsToOutput(
			List<Integer> amountsToRead,
			List<InputStream> inputStream,
			OutputStream outputStream
	) {
		BitWriterToFile writer = new BitWriterToFile(outputStream);
		int lastValue = 0;
		for (int i = 0; i < amountsToRead.size(); i++) {
			lastValue = addContentToFile(writer, inputStream.get(i), amountsToRead.get(i), lastValue);
		}
		writer.writeFinalResult();
	}

	private int addContentToFile(BitWriterToFile writer, InputStream inputStream, int amountToRead, int lastValue) {
		BitReaderBase reader = new BitReaderFromFile(inputStream);
		for (int i = 0; i < amountToRead; i++) {
			Integer number = this.getValue(reader, lastValue);
			this.encodeSingleNumber(writer, number, lastValue);
			lastValue = number;
		}
		return lastValue;
	}

	public static abstract class BitReaderBase {
		byte currentByte;
		int currentByteIndex;
		int currentIndex;
		boolean done;

		public BitReaderBase() {
			this.currentIndex = 0;
			this.currentByteIndex = 0;
			this.done = false;
			this.currentByte = 0;
		}

		protected abstract byte readNextByte();

		public Integer readNextNumber() {
			int offsetSize = 0;
			while (isOne()) {
				offsetSize++;
				incrementCurrentIndex();
			}
			incrementCurrentIndex(); // Reading the separator "0"
			return getInteger(offsetSize);
		}

		public Integer getInteger(int offsetSize) {
			int value = 1;
			for (int i = 0; i < offsetSize; i++) {
				if (isOne()) {
					value <<= 1;
					value |= 1;
				} else {
					value <<= 1;
				}
				incrementCurrentIndex();
			}
			return value;
		}

		public void incrementCurrentIndex() {
			currentIndex++;
			if (currentIndex == 8) {
				currentByteIndex++;
				if (!done) {
					currentByte = this.readNextByte();
				} else {
					currentByte = 0;
				}
				currentIndex = 0;
			}
		}

		public boolean isOne() {
			return (currentByte & (1 << currentIndex)) != 0;
		}
	}

	public static class BitReader extends BitReaderBase {

		private byte[] bytes;

		public BitReader(byte[] bytes) {
			this.bytes = bytes;
			this.currentByte = this.readNextByte();
		}

		@Override
		protected byte readNextByte() {
			if (this.currentByteIndex >= this.bytes.length) {
				this.done = true;
				return 0;
			} else {
				return this.bytes[this.currentByteIndex];
			}
		}
	}

	public static class BitReaderFromFile extends BitReaderBase {

		InputStream inputStream;

		public BitReaderFromFile(InputStream inputStream) {
			this(inputStream, 0);
		}

		public BitReaderFromFile(InputStream inputStream, int bufferSize) {
			if (bufferSize == 0) {
				this.inputStream = new BufferedInputStream(inputStream);
			} else {
				this.inputStream = new BufferedInputStream(inputStream, bufferSize);
			}
			this.currentByte = this.readNextByte();
		}

		@Override
		public byte readNextByte() {
			try {
				int read = inputStream.read();
				if (read != -1) {
					return (byte) read;
				} else {
					done = true;
					return 0;
				}
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException("readNextByte: error");
			}
		}

	}

	public static class BitWriter {
		List<Byte> result = new ArrayList<>();
		byte currentByte = 0;
		int currentIndex = 0;
		int byteCount = 0;

		void writeUnary(int len) {
			for (int i = 0; i < len; i++) {
				currentByte |= (byte) (1 << currentIndex);
				incrementCurrentIndex();
			}
		}

		protected void incrementCurrentIndex() {
			currentIndex++;
			if (currentIndex == 8) {
				byteCount++;
				this.writeCurrentByte();
				currentByte = 0;
				currentIndex = 0;
			}
		}

		protected void writeCurrentByte() {
			result.add(currentByte);
		}

		void writeBinary(int value, int size) {
			int mask = 1 << (size - 1);
			for (int i = 0; i < size; i++) {
				if ((mask & value) != 0) {
					this.writeUnary(1);
				} else {
					this.incrementCurrentIndex();
				}
				mask >>= 1;
			}
		}

		public List<Byte> getFinalResult() {
			if (currentIndex > 0) {
				result.add(currentByte);
			}
			return result;
		}

		public int getCurrentByteCount() {
			int add = currentIndex > 0 ? 1 : 0;
			return byteCount + add;
		}

		public int getCurrentBitCount() {
			return this.getCurrentByteCount() * 8 + this.currentIndex;
		}
	}

	public static class BitWriterToFile extends BitWriter {
		private final OutputStream outputStream;

		public BitWriterToFile(OutputStream outputStream) {
			this(outputStream, 0);
		}

		public BitWriterToFile(OutputStream outputStream, int bufferSize) {
			if (bufferSize == 0) {
				this.outputStream = new BufferedOutputStream(outputStream);
			} else {
				this.outputStream = new BufferedOutputStream(outputStream, bufferSize);
			}
		}

		protected void writeCurrentByte() {
			try {
				this.outputStream.write(currentByte);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		public void writeFinalResult() {
			this.writeFinalResult(true);
		}

		public void writeFinalResult(boolean toClose) {
			if (currentIndex > 0) {
				this.currentIndex = 7;
				this.incrementCurrentIndex(); // Will write the last byte and reset
			}
			if (toClose) {
				try {
					this.outputStream.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}
}
