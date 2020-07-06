package webdata.compression;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

public interface UnaryCoder {

	int getValue(GammaCode.BitReaderBase bitReader, Integer lastValue);

	List<Byte> encode(List<Integer> integerList);

	void encodeSingleNumber(GammaCode.BitWriter bitWriter, Integer number, Integer lastValue);

	List<Integer> decode(int amountToRead, byte[] bytes);

	void mergeTwoInputToOutput(int amountToRead1, int amountToRead2, InputStream inputStream1, InputStream inputStream2, OutputStream outputStream);

	void mergeInputsToOutput(List<Integer> amountsToRead, List<InputStream> inputStream, OutputStream outputStream);

	default void mergeTwoInputToOutputWithMap(int amountToRead1, int amountToRead2,
											  InputStream inputStream1, InputStream inputStream2,
											  List<Integer> map1, List<Integer> map2,
											  OutputStream outputStream) {
		throw new RuntimeException("mergeTwoInputToOutputWithMap");
	}

	default void mergeInputToOutputWithMaps(
			List<Integer> amountsToRead,
			List<InputStream> inputStreams,
			List<List<Integer>> maps,
			OutputStream outputStream) {
		throw new RuntimeException("mergeInputToOutputWithMaps");
	}
}
