package com.ppp.dataminer.nlp.topicmodel.plsa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.topicmodel.data.ScoreComparable;

/**
 * PLSA应用实现
 * 
 * @author zhimatech
 *
 */
public class PLSAInference {
	// 迭代次数
	private int iters = 100;

	private PLSAModel plsa = new PLSAModel();

	public void initializeModel(String modelPath) {
		plsa.initializeModel(modelPath);
	}

	public float[] plsaInference(String newDoc) {
		// wordIndex--count
		Map<Integer, Integer> termCountMap = new HashMap<Integer, Integer>();

		float[] topicPros = randomProbilities(plsa.getTopicTermPros().length);

		int wordCount = calcWordCount(newDoc, termCountMap, plsa.getWordDic());
		// 词列表，实际上是索引列表，这里的索引是指词在词典中的索引
		List<Integer> wordList = new ArrayList<Integer>(termCountMap.keySet());
		// p(z|d,w),由于这里d固定为应用文本，式子可以简化用二维数组表示
		float[][] docTermTopicPros = new float[termCountMap.size()][plsa.getTopicTermPros().length];
		// 迭代计算
		for (int iterCount = 0; iterCount < iters; iterCount++) {
			/*
			 * E步 计算p(z|d,w)
			 */
			for (int i = 0; i < wordList.size(); i++) {
				float sum = 0;
				float[] preTopicPro = new float[plsa.getTopicTermPros().length];
				for (int j = 0; j < plsa.getTopicTermPros().length; j++) {
					preTopicPro[j] = topicPros[j] * plsa.getTopicTermPros()[j][wordList.get(i)];
					sum += preTopicPro[j];
				}
				if (sum == 0) {
					sum = 0.000000001f;
				}

				for (int j = 0; j < plsa.getTopicTermPros().length; j++) {
					docTermTopicPros[i][j] = preTopicPro[j] / sum;
				}
			}

			/*
			 * M步 更新p(z|d)
			 */
			for (int j = 0; j < plsa.getTopicTermPros().length; j++) {
				float sum = 0;
				for (int i = 0; i < wordList.size(); i++) {
					sum += (float) termCountMap.get(wordList.get(i)) * docTermTopicPros[i][j];
				}
				topicPros[j] = sum / wordCount;
			}
		}

		return topicPros;
	}

	public List<String> getKeywords(float[] topicPros) {
		List<String> keywords = new ArrayList<String>();
		// 主题索引
		Integer[] index = new Integer[topicPros.length];
		for (int i = 0; i < index.length; i++) {
			index[i] = i;
		}
		Arrays.sort(index, new ScoreComparable(topicPros));

		for (int i = 0; i < 3; i++) {
			// 取前3个主题的前3个词
			List<String> wordList = plsa.getTopicKeywords().get(index[i]);
			for (int j = 0; j < 3; j++) {
				keywords.add(wordList.get(j));
			}
		}

		return keywords;
	}

	public float[] simplePlsaInference(String newDoc) {
		float[] topicPros = new float[plsa.getTopicTermPros().length];
		List<Term> parse = ToAnalysis.parse(newDoc);
		List<Integer> indexList = new ArrayList<Integer>();
		for (Term term : parse) {
			if (plsa.getWordDic().contains(term.getName())) {
				int index = plsa.getWordDic().indexOf(term.getName());
				indexList.add(index);
			}
		}

		float sum = 0;
		for (int topicIndex = 0; topicIndex < topicPros.length; topicIndex++) {
			topicPros[topicIndex] = 1;
			for (Integer wordIndex : indexList) {
				topicPros[topicIndex] *= plsa.getTopicTermPros()[topicIndex][wordIndex];
			}
			sum += topicPros[topicIndex];
		}

		if (sum == 0) {
			sum = 1;
		}

		for (int topicIndex = 0; topicIndex < topicPros.length; topicIndex++) {
			topicPros[topicIndex] /= sum;
		}

		return topicPros;
	}

	private int calcWordCount(String newDoc, Map<Integer, Integer> termCountMap, List<String> wordDic) {
		int wordCount = 0;
		List<Term> parse = ToAnalysis.parse(newDoc);
		for (Term term : parse) {
			if (wordDic.contains(term.getName())) {
				int index = wordDic.indexOf(term.getName());
				if (index < 0) {
					continue;
				}
				wordCount++;
				if (termCountMap.containsKey(index)) {
					termCountMap.put(index, termCountMap.get(index) + 1);
				} else {
					termCountMap.put(index, 1);
				}
			}
		}
		return wordCount;
	}

	private float[] randomProbilities(int size) {
		if (size < 1) {
			throw new IllegalArgumentException("The size param must be greate than zero");
		}
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
}
