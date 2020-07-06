package webdata.utils;

import java.util.ArrayList;
import java.util.List;

public class BitWriter {
	List<Byte> result = new ArrayList<>();
	byte currentByte = 0;
	int currentIndex = 0;

	public void writeUnary(int len) {
		for (int i = 0; i < len; i++) {
			currentByte |= (byte) (1 << currentIndex);
			incrementCurrentIndex();
		}
	}

	public  void incrementCurrentIndex() {
		currentIndex++;
		if (currentIndex == 8) {
			result.add(currentByte);
			currentByte = 0;
			currentIndex = 0;
		}
	}

	public void writeBinaryString(String binaryString) {
		for (int i = 0; i < binaryString.length(); i++) {
			if (binaryString.charAt(i) == '1') {
				this.writeUnary(1);
			} else {
				incrementCurrentIndex();
			}
		}
	}

	public List<Byte> getFinalResult() {
		if (currentIndex > 0) {
			result.add(currentByte);
		}
		return result;
	}
}
