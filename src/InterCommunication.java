import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class InterCommunication {
	public static BufferedReader inp;
	public static BufferedWriter out;

	public static void print(String s) {
		System.out.println(s);
	}

	public static String pipe(String msg) {
		String ret;

		try {
			out.write(msg + "\n");
			out.flush();
			ret = inp.readLine();
			return ret;
		} catch (Exception err) {

		}
		return "";
	}


	public static void main(String[] args) {
		try {
			Process p = Runtime.getRuntime().exec(new String[]{
					"python", "python_java_interprocess_communication/python_side.py"
			});
			inp = new BufferedReader(new InputStreamReader(p.getInputStream()));
			out = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));

			print(tagWord("az"));
			print(tagWord("RoteM"));
			terminatePythonSide();
			inp.close();
			out.close();
		} catch (Exception err) {
			err.printStackTrace();
		}
	}

	private static String tagWord(String word) {
		return pipe("tag:" + word);
	}

	private static void terminatePythonSide() {
		pipe("|");
	}
}
