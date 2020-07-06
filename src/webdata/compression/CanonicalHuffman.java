package webdata.compression;

import java.util.*;


public class CanonicalHuffman {
	public static final Character SEP = ' ';
	private final List<Character> alphaBet;
	Map<Character, Integer> counter;
	private TreeMap<Integer, TreeSet<Character>> lenToChars;
	private Map<Character, Integer> charToLen;
	private Map<Character, Integer> code;

	public CanonicalHuffman(List<Character> alphaBet) {
		this.alphaBet = alphaBet;
		lenToChars = null;
	}

	public static CanonicalHuffman createDefaultCanonicalHuffman() {
		List<Character> alphaBet = new ArrayList<>();
		for (char c = 'a'; c <= 'z'; c++) {
			alphaBet.add(c);
		}
		for (char c = '0'; c <= '9'; c++) {
			alphaBet.add(c);
		}
		return new CanonicalHuffman(alphaBet);
	}

	public void initCounter() {
		counter = new HashMap<>();
		for (Character character : alphaBet) {
			counter.put(character, 0);
		}

	}

	public void generateFrequency(String s) {
		for (int i = 0; i < s.length(); i++) {
			this.counter.compute(s.charAt(i), (a, b) -> b + 1);
		}
	}

	public void generateFrequency(List<String> sList) {
		for (String s : sList) {
			this.generateFrequency(s);
		}
	}

	public void buildHuffmanTree() {
		PQNode root = getHuffmanTree();
		lenToChars = this.buildCanonicalCode(root);
		counter.clear();
		counter = null;
		charToLen = getCharToLen();
		code = new HashMap<Character, Integer>();
		int cur_code = 0;
		int prevLen = 0;
		Iterator<Map.Entry<Integer, TreeSet<Character>>> iterator = lenToChars.entrySet().iterator();
		while (iterator.hasNext()) {
			Map.Entry<Integer, TreeSet<Character>> entry = iterator.next();
			int len = entry.getKey();
			cur_code <<= len - prevLen;
			prevLen = len;
			for (Iterator<Character> iter = entry.getValue().iterator(); iter.hasNext(); ) {
				char c = iter.next();
				if (!iterator.hasNext() && !iter.hasNext()) {
					cur_code = (1 << len) - 1;
				}
				code.put(c, cur_code);
				cur_code += 1;
			}
		}
	}

	private String padWithZeros(int value, int toPad) {
		String asBinary = Integer.toBinaryString(value);
		StringBuilder stringBuilder = new StringBuilder();
		for (int i = 0; i < toPad - asBinary.length(); i++) {
			stringBuilder.append('0');
		}
		stringBuilder.append(asBinary);
		return stringBuilder.toString();
	}

	private Map<Character, Integer> getCharToLen() {
		Map<Character, Integer> inverted = new HashMap<>();
		Set<Map.Entry<Integer, TreeSet<Character>>> entrySet = lenToChars.entrySet();
		for (Map.Entry<Integer, TreeSet<Character>> entry : entrySet) {
			Integer len = entry.getKey();
			for (Character c : entry.getValue()) {
				inverted.put(c, len);
			}
		}
		return inverted;
	}

	public byte[] encode(String input) {
		BitSet bitSet = new BitSet();
		int index = 0;
		for (int i = 0; i < input.length(); i++) {
			char c = input.charAt(i);
			if (!this.code.containsKey(c)) {
				int x = 1;
			}
			int code = this.code.get(c);
			int len = this.charToLen.get(c);
			for (int j = len - 1; j >= 0; j--) {
				boolean isOne = (code & (1 << j)) > 0;
				bitSet.set(index, isOne);
				index++;
			}
		}
		return bitSet.toByteArray();
	}

	public List<List<Byte>> getLenToChar() {
		Set<Map.Entry<Integer, TreeSet<Character>>> entries = this.lenToChars.entrySet();
		List<Integer> lensToSave = new ArrayList<>(this.alphaBet.size());
		List<Integer> charsToSave = new ArrayList<>(this.alphaBet.size());
		int lastLen = 0;
		for (Map.Entry<Integer, TreeSet<Character>> next : entries) {
			Integer len = next.getKey();
			for (; lastLen < len - 1; lastLen++) {
				lensToSave.add(0);
			}
			lastLen++;
			TreeSet<Character> characters = next.getValue();
			lensToSave.add(characters.size());
			for (Character c : characters) {
				charsToSave.add(Character.getNumericValue(c));
			}
		}
		for (int i = lastLen; i < alphaBet.size(); i++) {
			lensToSave.add(0);
		}
		ArrayList<List<Byte>> lists = new ArrayList<>();
		lists.add(new GammaCode(1).encode(lensToSave));
		lists.add(new GammaCode(1).encode(charsToSave));
		return lists;
	}

	public void loadLenToChar(byte[] lensAsBytes, byte[] charsSorted) {
		GammaCode gammaCode = new GammaCode(1);
		List<Integer> decode1 = gammaCode.decode(alphaBet.size(), charsSorted);
		List<Integer> decode2 = gammaCode.decode(alphaBet.size(), lensAsBytes);
		this.lenToChars = new TreeMap<>();
		int lastIndex = 0;
		for (int i = 0; i < decode2.size(); i++) {
			TreeSet<Character> treeSet = lenToChars.compute(i + 1, (integer, set) -> new TreeSet<>());
			for (int j = 0; j < decode2.get(i); j++) {
				Integer integer = decode1.get(lastIndex);
				treeSet.add(Character.forDigit(integer, 36));
				lastIndex++;
			}

		}
	}

	public String decode(byte[] data, int wordSize) {
		BitSet bitSet = BitSet.valueOf(data);
		StringBuilder stringBuilder = new StringBuilder();
		int bitSetIndex = 0;
		for (int i = 0; i < wordSize; i++) {
			Set<Map.Entry<Integer, TreeSet<Character>>> entries = this.lenToChars.entrySet();
			int c_i = 0;
			int d = 0;
			int lastLen = 0;
			for (Map.Entry<Integer, TreeSet<Character>> entry : entries) {
				int len = entry.getKey();
				for (int j = lastLen; j < len; j++) {
					d <<= 1;
					if (bitSetIndex < bitSet.length() && bitSet.get(bitSetIndex)) {
						d |= 1;
					}
					bitSetIndex++;
				}
				c_i <<= len - lastLen;
				lastLen = len;
				int a_i_size = entry.getValue().size();
				boolean cond = d <= c_i + a_i_size - 1;
				if (cond) {
					Iterator<Character> iterator = entry.getValue().iterator();
					int toSkip = d - c_i + 1;
					for (int j = 0; j < toSkip - 1; j++) {
						iterator.next();
					}
					stringBuilder.append(iterator.next());
					break;
				} else {
					c_i += a_i_size;
				}

			}
		}
		return stringBuilder.toString();
	}

	private PQNode getHuffmanTree() {
		Queue<PQNode> pqNodeQueue = new PriorityQueue<>(counter.size(), new Pq_compare());
		for (Map.Entry<Character, Integer> characterIntegerEntry : counter.entrySet()) {
			PQNode node = new PQNode();
			node.data = characterIntegerEntry.getValue();
			node.c = characterIntegerEntry.getKey();
			node.left = node.right = null;
			pqNodeQueue.add(node);
		}
		PQNode root = null;
		while (pqNodeQueue.size() > 1) {
			PQNode node1 = pqNodeQueue.peek();
			pqNodeQueue.poll();
			PQNode node2 = pqNodeQueue.peek();
			pqNodeQueue.poll();
			PQNode nodeobj = new PQNode();
			nodeobj.data = node1.data + node2.data;
			nodeobj.c = SEP;
			nodeobj.left = node1;
			nodeobj.right = node2;
			root = nodeobj;
			pqNodeQueue.add(nodeobj);
		}
		return root;
	}

	private TreeMap<Integer, TreeSet<Character>> buildCanonicalCode(PQNode root) {
		int codeLen = 0;
		Queue<PQNode> queue = new LinkedList<>();
		queue.add(root);
		TreeMap<Integer, TreeSet<Character>> result = new TreeMap<>();

		while (!queue.isEmpty()) {
			int size = queue.size();
			for (int i = 0; i < size; i++) {
				PQNode node = queue.peek();
				queue.poll();
				if (node.c != SEP) {
					TreeSet<Character> treeSet = result.computeIfAbsent(codeLen, (integer) -> new TreeSet<>());
					treeSet.add(node.c);
				}
				if (node.left != null) {
					queue.add(node.left);
				}
				if (node.right != null) {
					queue.add(node.right);
				}
			}
			codeLen += 1;

		}
		return result;
	}

	private static class PQNode {
		int data;
		char c;

		PQNode left;
		PQNode right;
	}

	static class Pq_compare implements Comparator<PQNode> {
		public int compare(PQNode a, PQNode b) {
			return a.data - b.data;
		}
	}


}
