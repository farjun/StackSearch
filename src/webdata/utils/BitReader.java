package webdata.utils;

public class BitReader {

	byte currentByte;
	int currentByteIndex;
	int currentIndex;
	private byte[] bytes;

	public BitReader(byte[] bytes) {
		this.bytes = bytes;
		this.currentIndex = 0;
		this.currentByteIndex = 0;
		this.currentByte = bytes[this.currentByteIndex];
	}

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
		StringBuilder numberAsBinary = new StringBuilder("1");
		for (int i = 0; i < offsetSize; i++) {
			if (isOne()) {
				numberAsBinary.append("1");
			} else {
				numberAsBinary.append("0");
			}
			incrementCurrentIndex();
		}
		return Integer.parseInt(numberAsBinary.toString(), 2);
	}

	private void incrementCurrentIndex() {
		currentIndex++;
		if (currentIndex == 8) {
			currentByteIndex++;
			if (currentByteIndex < this.bytes.length) {
				currentByte = this.bytes[currentByteIndex];
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
