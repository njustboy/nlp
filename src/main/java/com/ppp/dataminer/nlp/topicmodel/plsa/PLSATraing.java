package com.ppp.dataminer.nlp.topicmodel.plsa;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import com.ppp.dataminer.nlp.topicmodel.data.Documents;
import com.ppp.dataminer.nlp.topicmodel.data.ScoreComparable;
import com.ppp.dataminer.nlp.topicmodel.util.FileUtil;

/**
 * PLSA算法实现
 * 
 * @author zhimatech
 *
 */
public class PLSATraing {
	// 迭代次数
	private int iters = 100;
	// 迭代多少次输出一次结果
	private int saveStep = 10;
	// 迭代多少步后开始输出结果
	private int beginSaveIters = 50;
	// 主题数量
	private int topicNum = 100;
	// 文本数量
	private int N; // number of docs
	// 单词数量
	private int M; // number of terms
	// 文本--词矩阵，需要稀疏表示
	private int[][] docTermMatrix; // docTermMatrix
	// 文本--词集合 doc--词典索引--位置索引
	private Map<Integer, Integer>[] docTermMap;
	// 文本--主题 分布
	private float[][] docTopicPros;// p(z|d)
	// 主题--词 分布
	private float[][] topicTermPros;// p(w|z)
	// 文本，词--主题 分布，需要稀疏表示
	private float[][][] docTermTopicPros;// p(z|d,w)

	/**
	 * 使用格式化的语料初始化PLSA模型
	 * 
	 * @param docSet
	 */
	@SuppressWarnings("unchecked")
	public void initializeModel(Documents docSet) {
		if (docSet == null) {
			System.out.println("训练语料为空");
			System.exit(0);
		}

		// 文档数量
		N = docSet.getDocs().size();
		// 词数量
		M = docSet.getIndexToTermMap().size();

		docTermMatrix = new int[N][];
		docTermMap = new Map[N];
		// 初始化文本--词矩阵，矩阵稀疏表示
		for (int docIndex = 0; docIndex < N; docIndex++) {
			// 词典索引--count
			Map<Integer, Integer> map = merge(docSet.getDocs().get(docIndex).getDocWords());
			docTermMatrix[docIndex] = new int[map.size() * 2];
			int index = 0;
			Map<Integer, Integer> wordMap = new HashMap<Integer, Integer>();
			// 文本向量稀疏表示
			for (Entry<Integer, Integer> entry : map.entrySet()) {
				// 词索引
				docTermMatrix[docIndex][index++] = entry.getKey();
				// 词典索引--位置索引
				wordMap.put(entry.getKey(), index - 1);
				// 词频
				docTermMatrix[docIndex][index++] = entry.getValue();
			}
			// docTermMap.add(docIndex, wordMap);
			docTermMap[docIndex] = wordMap;
		}

		docTopicPros = new float[N][topicNum];
		// 初始化 p(z|d),每一个文档应该满足 sum(p(z|d))=1.0
		for (int i = 0; i < N; i++) {
			// 生成随机数向量，向量和为1
			float[] pros = randomProbilities(topicNum);
			for (int j = 0; j < topicNum; j++) {
				docTopicPros[i][j] = pros[j];
			}
		}

		topicTermPros = new float[topicNum][M];
		// 初始化 p(w|z),对于每一个主题应该满足 sum(p(w|z))=1.0
		for (int i = 0; i < topicNum; i++) {
			float[] pros = randomProbilities(M);
			for (int j = 0; j < M; j++) {
				topicTermPros[i][j] = pros[j];
			}
		}

		// 稀疏表示 p(z|d,w)
		docTermTopicPros = new float[N][][];
		for (int i = 0; i < N; i++) {
			docTermTopicPros[i] = new float[docTermMatrix[i].length / 2][topicNum];
		}

		// 写出词典，在应用的时候需要保证词典顺序是一致的
		FileUtil.writeLines("plsamodel/wordDic", docSet.getIndexToTermMap());
	}

	private Map<Integer, Integer> merge(int[] array) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i : array) {
			if (map.containsKey(i)) {
				map.put(i, map.get(i) + 1);
			} else {
				map.put(i, 1);
			}
		}
		return map;
	}

	/**
	 * 
	 * 使用EM算法训练模型
	 * 
	 * @param docs
	 *            all documents
	 * @throws IOException
	 */
	public void learnModel(Documents docSet) throws IOException {
		for (int i = 0; i < iters; i++) {
			System.out.println(" --------------第 " + i + " 次迭代-------------- ");
			if ((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)) {
				// 保存中间过程的模型
				System.out.println("保存第 " + i + " 次迭代训练的模型结果 ");
				saveIteratedModel(i, docSet);
			}
			// em();
			fastEM();
			System.out.println("似然函数值为："+computeLogLikelihood());
		}
	}

	/**
	 * 计算似然函数，注意这里把似然函数中的常量项去除了
	 * 
	 * @param docSet
	 * @return
	 */
	private double computeLogLikelihood() {
		double L = 0.0;
		for (int docIndex = 0; docIndex < N; docIndex++) {
			int docLength = docTermTopicPros[docIndex].length;
			for (int wordIndex = 0; wordIndex < docLength; wordIndex++) {
				double sumK = 0.0;
				int realWordIndex = docTermMatrix[docIndex][wordIndex << 1];
				for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
					sumK += docTopicPros[docIndex][topicIndex] * topicTermPros[topicIndex][realWordIndex];
				}
				L += (double) docTermMatrix[docIndex][(wordIndex<<1) + 1] * Math.log10(sumK);
			}
		}
		return L;
	}

	/**
	 * 
	 * EM algorithm
	 * 
	 */
	private void em() {
		/*
		 * E步，计算 p(z|d,w)
		 * 
		 * p(z|d,w)=p(z|d)*p(w|z)/sum(p(z'|d)*p(w|z'))
		 * 
		 */
		// 缓存公式中的分子
		// float[] perTopicPro = new float[topicNum];
		long timestamps1 = System.currentTimeMillis();
		for (int docIndex = 0; docIndex < N; docIndex++) {
			int length = docTermTopicPros[docIndex].length;
			for (int wordIndex = 0; wordIndex < length; wordIndex++) {
				// wordIndex需要映射至词在整个列表中的索引值
				int realWordIndex = docTermMatrix[docIndex][wordIndex << 1];
				float total = 0f;

				for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
					float numerator = docTopicPros[docIndex][topicIndex] * topicTermPros[topicIndex][realWordIndex];
					total += numerator;
					// perTopicPro[topicIndex] = numerator;
					docTermTopicPros[docIndex][wordIndex][topicIndex] = numerator;
				}

				if (total == 0.0) {
					total = avoidZero(total);
				}

				for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
					docTermTopicPros[docIndex][wordIndex][topicIndex] = docTermTopicPros[docIndex][wordIndex][topicIndex]
							/ total;
				}
			}
		}
		long timestamps2 = System.currentTimeMillis();
		System.out.println("E步计算p(z|d,w)耗时：" + (timestamps2 - timestamps1) / 1000 + "秒");

		// M步
		/*
		 * 更新 p(w|z) p(w|z)=sum(n(d',w)*p(z|d',w))/sum(sum(n(d',w')*p(z|d',w'))
		 */
		for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
			float totalDenominator = 0f;
			for (int wordIndex = 0; wordIndex < M; wordIndex++) {
				float numerator = 0f;
				for (int docIndex = 0; docIndex < N; docIndex++) {
					// 不是每一个doc都含有当前word，如果没有当前word则需要跳过
					// docterm中的索引
					// 词典索引--真实索引
					Map<Integer, Integer> wordMap = docTermMap[docIndex];
					// 真实索引
					Integer realIndex = wordMap.get(wordIndex);

					if (realIndex == null) {
						continue;
					}
					numerator += docTermMatrix[docIndex][realIndex + 1]
							* docTermTopicPros[docIndex][realIndex >> 1][topicIndex];
				}

				topicTermPros[topicIndex][wordIndex] = numerator;

				totalDenominator += numerator;
			}

			if (totalDenominator == 0.0) {
				totalDenominator = avoidZero(totalDenominator);
			}

			for (int wordIndex = 0; wordIndex < M; wordIndex++) {
				topicTermPros[topicIndex][wordIndex] = topicTermPros[topicIndex][wordIndex] / totalDenominator;
			}
		}
		long timestamps3 = System.currentTimeMillis();
		System.out.println("M步计算p(w|z)耗时：" + (timestamps3 - timestamps2) / 1000 + "秒");

		/*
		 *
		 * 更新p(z|d),p(z|d)=sum(n(d,w')*p(z|d,w'))/sum(sum(n(d,w')*p(z'|d,w')))
		 * 
		 * p(z|d)=sum(n(d,w')*p(z|d,w'))/n(d)
		 * 
		 */
		for (int docIndex = 0; docIndex < N; docIndex++) {
			// doc的词数量
			float totalDenominator = 0f;
			for (int i = 0; i < docTermMatrix[docIndex].length >> 1; i++) {
				totalDenominator += docTermMatrix[docIndex][(i << 1) + 1];
			}

			// 词典索引--位置索引
			Map<Integer, Integer> wordMap = docTermMap[docIndex];
			for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
				float numerator = 0;
				for (int wordIndex = 0; wordIndex < M; wordIndex++) {
					// 真实索引
					Integer realIndex = wordMap.get(wordIndex);

					if (realIndex == null) {
						continue;
					}
					numerator += docTermMatrix[docIndex][realIndex + 1]
							* docTermTopicPros[docIndex][realIndex >> 1][topicIndex];
				}

				docTopicPros[docIndex][topicIndex] = numerator / totalDenominator;
			}
		}

		long timestamps4 = System.currentTimeMillis();
		System.out.println("M步计算p(z|d)耗时：" + (timestamps4 - timestamps3) / 1000 + "秒");
	}

	private void fastEM() {
		/*
		 * E步，计算 p(z|d,w)
		 * 
		 * p(z|d,w)=p(z|d)*p(w|z)/sum(p(z'|d)*p(w|z'))
		 * 
		 */
		// 缓存公式中的分子
		// float[] perTopicPro = new float[topicNum];
		long timestamps1 = System.currentTimeMillis();
		for (int docIndex = 0; docIndex < N; docIndex++) {
			int length = docTermTopicPros[docIndex].length;
			for (int wordIndex = 0; wordIndex < length; wordIndex++) {
				// wordIndex需要映射至词在整个列表中的索引值
				int realWordIndex = docTermMatrix[docIndex][wordIndex << 1];
				float total = 0f;

				for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
					float numerator = docTopicPros[docIndex][topicIndex] * topicTermPros[topicIndex][realWordIndex];
					total += numerator;
					// perTopicPro[topicIndex] = numerator;
					docTermTopicPros[docIndex][wordIndex][topicIndex] = numerator;
				}

				if (total == 0.0) {
					total = avoidZero(total);
				}

				for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
					docTermTopicPros[docIndex][wordIndex][topicIndex] = docTermTopicPros[docIndex][wordIndex][topicIndex]
							/ total;
				}
			}
		}
		long timestamps2 = System.currentTimeMillis();
		System.out.println("E步计算p(z|d,w)耗时：" + (timestamps2 - timestamps1) / 1000 + "秒");

		// M步
		/*
		 * 更新 p(w|z) p(w|z)=sum(n(d',w)*p(z|d',w))/sum(sum(n(d',w')*p(z|d',w'))
		 */
		float[] tmpWord = null;
		for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
			float totalDenominator = 0f;
			tmpWord = new float[M];
			for (int docIndex = 0; docIndex < N; docIndex++) {
				// 这个doc的词的个数
				int wordLength = docTermTopicPros[docIndex].length;
				for (int wordIndex = 0; wordIndex < wordLength; wordIndex++) {
					tmpWord[docTermMatrix[docIndex][wordIndex << 1]] += docTermMatrix[docIndex][(wordIndex << 1) + 1]
							* docTermTopicPros[docIndex][wordIndex][topicIndex];
				}
			}

			// for (int docIndex = 0; docIndex < N; docIndex++) {
			// for (int wordIndex = 0; wordIndex < M; wordIndex++) {
			// tmpWord[wordIndex] += tmpDocWord[docIndex][wordIndex];
			// }
			// }
			for (int wordIndex = 0; wordIndex < M; wordIndex++) {
				topicTermPros[topicIndex][wordIndex] = tmpWord[wordIndex];
				totalDenominator += tmpWord[wordIndex];
			}

			if (totalDenominator == 0.0) {
				totalDenominator = avoidZero(totalDenominator);
			}

			for (int wordIndex = 0; wordIndex < M; wordIndex++) {
				topicTermPros[topicIndex][wordIndex] = topicTermPros[topicIndex][wordIndex] / totalDenominator;
			}
		}
		long timestamps3 = System.currentTimeMillis();
		System.out.println("M步计算p(w|z)耗时：" + (timestamps3 - timestamps2) / 1000 + "秒");

		/*
		 *
		 * 更新p(z|d),p(z|d)=sum(n(d,w')*p(z|d,w'))/sum(sum(n(d,w')*p(z'|d,w')))
		 * 
		 * p(z|d)=sum(n(d,w')*p(z|d,w'))/n(d)
		 * 
		 */
		for (int docIndex = 0; docIndex < N; docIndex++) {
			// doc的词数量
			float totalDenominator = 0f;
			for (int i = 1; i < docTermMatrix[docIndex].length; i += 2) {
				totalDenominator += docTermMatrix[docIndex][i];
			}

			for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
				float numerator = 0;
				int wordLength = docTermTopicPros[docIndex].length;
				for (int wordIndex = 0; wordIndex < wordLength; wordIndex++) {
					numerator += docTermMatrix[docIndex][(wordIndex << 1) + 1]
							* docTermTopicPros[docIndex][wordIndex][topicIndex];
				}

				docTopicPros[docIndex][topicIndex] = numerator / totalDenominator;
			}
		}

		long timestamps4 = System.currentTimeMillis();
		System.out.println("M步计算p(z|d)耗时：" + (timestamps4 - timestamps3) / 1000 + "秒");
	}

	/**
	 * 
	 * 
	 * Get a normalize array
	 * 
	 * @param size
	 * @return
	 */
	public float[] randomProbilities(int size) {
		float[] pros = new float[size];

		int total = 0;
		Random r = new Random();
		for (int i = 0; i < pros.length; i++) {
			// avoid zero
			pros[i] = r.nextInt(size) + 1;

			total += pros[i];
		}

		// normalize
		for (int i = 0; i < pros.length; i++) {
			pros[i] = pros[i] / total;
		}

		return pros;
	}

	/**
	 * 
	 * @return
	 */
	public float[][] getDocTopics() {
		return docTopicPros;
	}

	/**
	 * 
	 * @return
	 */
	public float[][] getTopicWordPros() {
		return topicTermPros;
	}

	/**
	 * 
	 * Get topic number
	 * 
	 * 
	 * @return
	 */
	public Integer getTopicNum() {
		return topicNum;
	}

	/**
	 * 
	 * avoid zero number.if input number is zero, we will return a magic number.
	 * 
	 * 
	 */
	private final static float MAGICNUM = 0.000000001f;

	public float avoidZero(float num) {
		if (num == 0.0) {
			return MAGICNUM;
		}

		return num;
	}

	public void saveIteratedModel(int iteration, Documents docSet) throws IOException {
		String resPath = "plsamodel/model_fast_" + iteration;
		FileUtil.write2DArray(docTopicPros, resPath + ".docTopicPros");
		FileUtil.write2DArray(topicTermPros, resPath + ".topicTermPros");

		int topNum = 100;
		// 输出每个主题的topN词汇
		ArrayList<String> ztermsLines = new ArrayList<String>();
		for (int i = 0; i < topicNum; i++) {
			List<Integer> tWordsIndexArray = new ArrayList<Integer>();
			for (int w = 0; w < M; w++) {
				tWordsIndexArray.add(new Integer(w));
			}
			Collections.sort(tWordsIndexArray, new ScoreComparable(topicTermPros[i]));
			String line = "topic=" + i + "\t";
			for (int w = 0; w < topNum; w++) {
				line += docSet.getIndexToTermMap().get(tWordsIndexArray.get(w)) + "\t";
			}
			ztermsLines.add(line);
		}
		FileUtil.writeLines(resPath + ".zterms", ztermsLines);
	}
}
