package webdata.field_encoder_decoder.productId;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ProductIdMerger {

	public static int merge(String inDir1, int amountOfReviews1, String inDir2, int amountOfReviews2, String outDir) {
		int amountOfTokens = 0;
		try {
			ProductIdCompressor.InputStreamWrapper wrapper1 = new ProductIdCompressor.InputStreamWrapper(inDir1);
			ProductIdCompressor.InputStreamWrapper wrapper2 = new ProductIdCompressor.InputStreamWrapper(inDir2);
			ProductIdCompressor.OutputStreamWrapper outputStreamWrapper = new ProductIdCompressor.OutputStreamWrapper(outDir);
			ProductIdCompressor textCompressor = new ProductIdCompressor();
			amountOfTokens = textCompressor.mergeInputsToOutput(wrapper1, amountOfReviews1,
					wrapper2, amountOfReviews2,
					outputStreamWrapper);
			wrapper1.close();
			wrapper2.close();
			outputStreamWrapper.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return amountOfTokens;
	}


	public static int megaMerge(List<Integer> amountOfReviews, List<String> inDirs, String outDir) {
		int amountOfTokens = 0;
		try {
			List<ProductIdCompressor.InputStreamWrapper> l = new ArrayList<>();
			for (String inDir : inDirs) {
				l.add(new ProductIdCompressor.InputStreamWrapper(inDir));
			}
			ProductIdCompressor.OutputStreamWrapper outputStreamWrapper = new ProductIdCompressor.OutputStreamWrapper(outDir);
			ProductIdCompressor textCompressor = new ProductIdCompressor();
			amountOfTokens = textCompressor.megaMerge(l, amountOfReviews, outputStreamWrapper);
			for (ProductIdCompressor.InputStreamWrapper inputStreamWrapper : l) {
				inputStreamWrapper.close();
			}
			outputStreamWrapper.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return amountOfTokens;
	}
}
