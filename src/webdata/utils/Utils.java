package webdata.utils;

import webdata.compression.DeltaCode;
import webdata.compression.DeltaCodeGap;
import webdata.compression.GammaCode;
import webdata.compression.GammaCodeGap;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

public abstract class Utils {

	public static byte[] readFromPosition(FileInputStream fileInputStream, int lengthToRead, int position) {
		try {
			byte[] buf = new byte[lengthToRead];
			fileInputStream.getChannel().position(position);
			fileInputStream.read(buf);
			return buf;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return new byte[0];
	}

	public static byte[] readFromPositionRA(RandomAccessFile randomAccessFile, int lengthToRead, int position) {
		byte[] buf = new byte[lengthToRead];
		try {
			randomAccessFile.seek(position);
			randomAccessFile.read(buf);
		} catch (IOException ignored) {
		}
		return buf;

	}

	public static byte[] readFile(String path) {
		try {
			File file = new File(path);
			FileInputStream fileInputStream = new FileInputStream(file);
			byte[] data = new byte[(int) file.length()];
			fileInputStream.read(data);
			fileInputStream.close();
			return data;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	public static void writeFile(String path, byte[] data) {
		File file = new File(path);
		file.getParentFile().mkdirs();
		try {
			FileOutputStream fileOutputStream = new FileOutputStream(file);
			fileOutputStream.write(data);
			fileOutputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void writeFile(String path, List<Byte> data) {
		Utils.writeFile(path, convertToByteArray(data));
	}

	public static byte[] convertToByteArray(List<Byte> byteList) {
		byte[] bytes = new byte[byteList.size()];
		for (int i = 0; i < byteList.size(); i++) {
			bytes[i] = byteList.get(i);
		}
		return bytes;
	}

	public static byte[] intToByteArray(int value) {
		return ByteBuffer.allocate(4).putInt(value).array();
	}

	public static int byteArrayToInt(byte[] value, int index) {
		return ByteBuffer.wrap(value).getInt(index);
	}

	public static void addLastValueForEachM(List<Integer> integers, int steps) {
		int lastValue = 0;
		for (int i = 0; i < integers.size(); i += steps) {
			int value = integers.get(i) + lastValue;
			integers.set(i, value);
			lastValue = value;
		}

	}

	public static void compareCompression(String title, List<Integer> list, boolean useGap, int addForEach) {
		System.out.println("===== " + title + " =====");
		System.out.println("CompareCompression");
		System.out.println("GammaCode:" + new GammaCode(addForEach).encode(list).size());
		if (useGap) {
			System.out.println("GammaCodeGap:" + new GammaCodeGap(addForEach).encode(list).size());
		}
		System.out.println("DeltaCode:" + new DeltaCode(addForEach).encode(list).size());
		if (useGap) {
			System.out.println("DeltaCodeGap:" + new DeltaCodeGap(addForEach).encode(list).size());
		}
	}

	public static void compareCompression(String title, List<Integer> list, int addForEach) {
		compareCompression(title, list, false, addForEach);
	}

	public static void compareCompression(String title, List<Integer> list, boolean useGap) {
		compareCompression(title, list, useGap, 0);
	}

	public static void compareCompression(String title, List<Integer> list) {
		compareCompression(title, list, false, 0);
	}

	public static void copyFolder(File source, File destination) {
		if (!destination.exists()) {
			destination.mkdirs();
		}
		String[] files = source.list();
		for (String file : files) {
			File srcFile = new File(source, file);
			File destFile = new File(destination, file);
			copyFolderRec(srcFile, destFile);
		}
	}

	public static void copyFolderRec(File source, File destination) {
		if (source.isDirectory()) {
			return;
		}
		InputStream in = null;
		OutputStream out = null;
		try {
			in = new FileInputStream(source);
			out = new FileOutputStream(destination);
			byte[] buffer = new byte[1024];
			int length;
			while ((length = in.read(buffer)) > 0) {
				out.write(buffer, 0, length);
			}
			in.close();
			out.close();
		} catch (Exception e) {
			try {
				in.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}

			try {
				out.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
	}


	public static class CollectionEnumeration implements Enumeration<Integer> {

		private final Iterator<Integer> iterator;

		public CollectionEnumeration(Collection<Integer> list) {
			if (list != null) {
				iterator = list.iterator();
			} else {
				iterator = Collections.emptyIterator();
			}
		}

		/**
		 * Tests if this enumeration contains more elements.
		 *
		 * @return <code>true</code> if and only if this enumeration object
		 * contains at least one more element to provide;
		 * <code>false</code> otherwise.
		 */
		@Override
		public boolean hasMoreElements() {
			return this.iterator.hasNext();
		}

		/**
		 * Returns the next element of this enumeration if this enumeration
		 * object has at least one more element to provide.
		 *
		 * @return the next element of this enumeration.
		 * @throws NoSuchElementException if no more elements exist.
		 */
		@Override
		public Integer nextElement() {
			return iterator.next();
		}
	}

}
