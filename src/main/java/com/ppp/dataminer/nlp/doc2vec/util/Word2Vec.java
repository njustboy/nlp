package com.ppp.dataminer.nlp.doc2vec.util;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;

public class Word2Vec {
	private HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	// 相关词表
	private Map<String, List<String>> wordpairMap = new HashMap<String, List<String>>();
	//
	private static final int MAX_SIZE = 50;
	// 单例
	private volatile static Word2Vec instance = null;
	// word2vec训练度词个数
	private int words;
	// 词向量长度
	private int size;

	private int topNSize = 10;

	private Word2Vec() {
		loadGoogleModel("model/");
	}

	public static Word2Vec getInstance() {
		if (instance == null) {
			synchronized (Word2Vec.class) {
				instance = new Word2Vec();
			}
		}
		return instance;
	}

	/**
	 * 加载模型 模型由google的word2vec工具生成
	 * 
	 * @param path
	 *            模型的路径
	 */
	public void loadGoogleModel(String path) {
		DataInputStream dis = null;
		BufferedInputStream bis = null;
		BufferedReader br = null;
		double len = 0;
		float vector = 0;
		try {
			bis = new BufferedInputStream(
					new FileInputStream(new File("/Users/zhimatech/Desktop/word2vec-master/data/factorys.bin")));
			dis = new DataInputStream(bis);
			// //读取词数
			words = Integer.parseInt(readString(dis));
			// //大小
			size = Integer.parseInt(readString(dis));
			String word;
			float[] vectors = null;
			for (int i = 0; i < words; i++) {
				word = readString(dis);
				vectors = new float[size];
				len = 0;
				// 计算向量的模
				for (int j = 0; j < size; j++) {
					vector = readFloat(dis);
					// len += vector * vector;
					vectors[j] = (float) vector;
				}
				// 向量单位化
				// len = Math.sqrt(len);
				// for (int j = 0; j < size; j++) {
				// vectors[j] /= len;
				// }
				// 词向量存储
				wordMap.put(word, vectors);
				dis.read();
			}

			// 加载同义词
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(path + "simwords")), "UTF-8"));
			String line = br.readLine();
			while (line != null) {
				String[] pairs = line.split(":");
				if (pairs.length != 2) {
					line = br.readLine();
					continue;
				}
				String[] simwordsArr = pairs[1].split("\\s+");
				wordpairMap.put(pairs[0], Arrays.asList(simwordsArr));
				line = br.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (dis != null) {
				try {
					dis.close();
				} catch (Exception e) {

				}
			}
		}
	}

	/**
	 * 加载模型,模型由本工程的java代码生存
	 * 
	 * @param path
	 *            模型的路径
	 */
	public void loadJavaModel(String path) {
		DataInputStream dis = null;
		try {
			dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
			words = dis.readInt();
			size = dis.readInt();

			float vector = 0;

			String key = null;
			float[] value = null;
			for (int i = 0; i < words; i++) {
				double len = 0;
				key = dis.readUTF();
				value = new float[size];
				for (int j = 0; j < size; j++) {
					vector = dis.readFloat();
					len += vector * vector;
					value[j] = vector;
				}

				len = Math.sqrt(len);

				for (int j = 0; j < size; j++) {
					value[j] /= len;
				}
				wordMap.put(key, value);
			}
		} catch (Exception e) {
		} finally {
			if (dis != null) {
				try {
					dis.close();
				} catch (Exception e) {

				}
			}
		}
	}

	/**
	 * 返回两个给定单词的相似度
	 * 
	 * @param word1
	 * @param word2
	 * @return
	 */
	public float getDistance(String word1, String word2) {
		float distance = 0;
		if (word1.equalsIgnoreCase(word2)) {
			return 1;
		}
		if (!wordMap.containsKey(word1) || !wordMap.containsKey(word2)) {
			return 0;
		}
		float[] vector1 = wordMap.get(word1);
		float[] vector2 = wordMap.get(word2);
		int length = vector1.length < vector2.length ? vector1.length : vector2.length;

		for (int i = 0; i < length; i++) {
			distance += vector1[i] * vector2[i];
		}

		return distance;
	}

	/**
	 * 计算两个带权值词列表的相似度 通过将词向量加权平均后计算点积实现
	 * 
	 * @param list1
	 * @param list2
	 * @return
	 */
	public float getSimility(List<WordPair> list1, List<WordPair> list2) {
		float distance = 0;
		if (list1 == null || list2 == null || list1.size() == 0 || list2.size() == 0) {
			return distance;
		}
		float[] vector1 = new float[size];
		float[] vector2 = new float[size];
		int count = 0;
		for (WordPair wp : list1) {
			if (wordMap.containsKey(wp.getWord())) {
				float[] tmpVector = vectorMultiply(wordMap.get(wp.getWord()), wp.getWeight());
				vector1 = vectorAdd(vector1, tmpVector);
				count++;
			}
		}
		if (count > 0) {
			vector1 = vectorMultiply(vector1, 1.0 / count);
		}

		count = 0;
		for (WordPair wp : list2) {
			if (wordMap.containsKey(wp.getWord())) {
				float[] tmpVector = vectorMultiply(wordMap.get(wp.getWord()), wp.getWeight());
				vector2 = vectorAdd(vector2, tmpVector);
				count++;
			}
		}
		if (count > 0) {
			vector2 = vectorMultiply(vector2, 1.0 / count);
		}

		int minLength = vector1.length > vector2.length ? vector2.length : vector1.length;
		int maxLength = vector1.length + vector2.length - minLength;
		for (int i = 0; i < minLength; i++) {
			distance += vector1[i] * vector2[i];
		}

		distance *= Math.pow((double) minLength / maxLength, 0.5);

		return distance;
	}

	/**
	 * 向量相加
	 * 
	 * @param vector1
	 * @param vector2
	 * @return
	 */
	private static float[] vectorAdd(float[] vector1, float[] vector2) {
		int minLength = vector1.length < vector2.length ? vector1.length : vector2.length;
		for (int i = 0; i < minLength; i++) {
			vector1[i] += vector2[i];
		}
		return vector1;
	}

	/**
	 * 向量乘以常数
	 * 
	 * @param vector
	 * @param d
	 * @return
	 */
	private static float[] vectorMultiply(float[] vector, double d) {
		for (int i = 0; i < vector.length; i++) {
			vector[i] *= d;
		}
		return vector;
	}

	/**
	 * 通过两个词靠近的距离判断其相似度
	 * 
	 * @param word1
	 * @param word2
	 * @return
	 */
	public float getSimility(String word1, String word2) {
		float simility = 0f;
		if (word1.equalsIgnoreCase(word2)) {
			return 1;
		}
		if (!wordpairMap.containsKey(word1) || !wordpairMap.containsKey(word2)) {
			return simility;
		}
		float sim1 = 0;
		List<String> simWords1 = wordpairMap.get(word1);
		for (int i = 0; i < simWords1.size(); i++) {
			if (simWords1.get(i).equals(word2)) {
				sim1 = 1.0f * (simWords1.size() - i) / simWords1.size();
				break;
			}
		}
		float sim2 = 0;
		List<String> simWords2 = wordpairMap.get(word2);
		for (int i = 0; i < simWords2.size(); i++) {
			if (simWords2.get(i).equals(word1)) {
				sim2 = 1.0f * (simWords2.size() - i) / simWords2.size();
				break;
			}
		}

		return (sim1 + sim2) / 2;
	}

	/**
	 * 返回float
	 * 
	 * @param is
	 * @return
	 */
	public static float readFloat(InputStream is) {
		byte[] bytes = new byte[4];
		try {
			is.read(bytes);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return getFloat(bytes);
	}

	/**
	 * 读取一个float
	 * 
	 * @param b
	 * @return
	 */
	public static float getFloat(byte[] b) {
		int accum = 0;
		// 按照特定的存储方式读取数据
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}

	/**
	 * 读取一个字符串
	 * 
	 * @param dis
	 * @return
	 */
	private static String readString(DataInputStream dis) {
		byte[] bytes = new byte[MAX_SIZE];
		int i = -1;
		StringBuilder sb = new StringBuilder();
		try {
			byte b = dis.readByte();
			// 按照特定的格式读取文件；32与10分别表示空格和换行。
			while (b != 32 && b != 10) {
				i++;
				bytes[i] = b;
				b = dis.readByte();
				if (i == MAX_SIZE - 1) {
					sb.append(new String(bytes, "UTF-8"));
					i = -1;
					bytes = new byte[MAX_SIZE];
				}
			}
			sb.append(new String(bytes, 0, i + 1, "UTF-8"));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return sb.toString();
	}

	/**
	 * 近义词
	 * 
	 * @return
	 */
	public TreeSet<WordPair> analogy(String word0, String word1, String word2) {
		float[] wv0 = getWordVector(word0);
		float[] wv1 = getWordVector(word1);
		float[] wv2 = getWordVector(word2);

		if (wv1 == null || wv2 == null || wv0 == null) {
			return null;
		}
		float[] wordVector = new float[size];
		for (int i = 0; i < size; i++) {
			wordVector[i] = wv1[i] - wv0[i] + wv2[i];
		}
		float[] tempVector;
		String name;
		List<WordPair> wordEntrys = new ArrayList<WordPair>(topNSize);
		for (Entry<String, float[]> entry : wordMap.entrySet()) {
			name = entry.getKey();
			if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
				continue;
			}
			float dist = 0;
			tempVector = entry.getValue();
			for (int i = 0; i < wordVector.length; i++) {
				dist += wordVector[i] * tempVector[i];
			}
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<WordPair>(wordEntrys);
	}

	private void insertTopN(String name, float score, List<WordPair> wordsEntrys) {
		// TODO Auto-generated method stub
		if (wordsEntrys.size() < topNSize) {
			wordsEntrys.add(new WordPair(name, score));
			return;
		}
		float min = Float.MAX_VALUE;
		int minOffe = 0;
		for (int i = 0; i < topNSize; i++) {
			WordPair wordEntry = wordsEntrys.get(i);
			if (min > wordEntry.getWeight()) {
				min = (float) wordEntry.getWeight();
				minOffe = i;
			}
		}

		if (score > min) {
			wordsEntrys.set(minOffe, new WordPair(name, score));
		}

	}

	public Set<WordPair> distance(String queryWord) {

		float[] center = wordMap.get(queryWord);
		if (center == null) {
			return Collections.emptySet();
		}

		int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		TreeSet<WordPair> result = new TreeSet<WordPair>();

		double norm = 0;
		for (int i = 0; i < center.length; i++) {
			norm += center[i] * center[i];
		}
		norm = Math.sqrt(norm);

		double min = Float.MIN_VALUE;

		for (Map.Entry<String, float[]> entry : wordMap.entrySet()) {
			float[] vector = entry.getValue();
			float dist = 0;
			for (int i = 0; i < vector.length; i++) {
				dist += center[i] * vector[i];
			}
			double norm1 = 0;
			for (int i = 0; i < vector.length; i++) {

				norm1 += vector[i] * vector[i];
			}
			norm1 = Math.sqrt(norm1);
			dist = (float) (dist / (norm * norm1));
			// if (dist > min) {
			result.add(new WordPair(entry.getKey(), dist));
			if (resultSize < result.size()) {
				result.pollLast();
			}
			min = result.last().getWeight();
			// }
		}
		result.pollFirst();// 本身

		return result;
	}

	/**
	 * 得到词向量
	 * 
	 * @param word
	 * @return
	 */
	public float[] getWordVector(String word) {
		return wordMap.get(word);
	}

	/**
	 * 获得词典集合
	 * 
	 * @return
	 */
	public Set<String> getWords() {
		return wordMap.keySet();
	}
}
