package com.ppp.dataminer.nlp.topicmodel.lda;

import java.util.ArrayList;
import java.util.List;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.topicmodel.util.FileUtil;

/**
 * LDA模型应用
 * 
 * @author zhimatech
 *
 */
public class LDAInference {
	// 词典列表
	List<String> wordDic = new ArrayList<String>();
	// 迭代计算次数
	int iterations = 100;
	// usual value is 50 / K
	float alpha = 0.5f; // doc-topic dirichlet prior parameter
	// usual value is 0.01
	float beta = 0.01f; // topic-word dirichlet prior parameter
	float[][] phi;// Parameters for topic-word distribution K*V
	// 主题数量
	int K;
	// 不同主题下的词数量
	int[] wordCount;

	/**
	 * 使用模型文件进行初始化
	 * 
	 * @param dicPath
	 */
	public void initializeModel(String dicPath) {
		// 读入词典列表
		FileUtil.readLines("ldamodel/wordDic", wordDic);
		// 从模型文件中读入 topic-word 概率矩阵
		phi = FileUtil.read2DArray("ldamodel/lda_100.phi");

		K = phi.length;
		
		alpha = 50/K;
	}

	public float[] ldaInference(String newDoc) {
		float[] topicVec = new float[K];
		List<Term> parse = ToAnalysis.parse(newDoc);
		// 词列表
		List<Integer> words = new ArrayList<Integer>();
		for (Term term : parse) {
			if (wordDic.contains(term.getName())) {
				words.add(wordDic.indexOf(term.getName()));
			}
		}

		// 为每个词初始化一个主题
		int[] wordTopics = new int[words.size()];
		for (int i = 0; i < wordTopics.length; i++) {
			wordTopics[i] = (int) (Math.random() * K);
		}

		// 每个主题下词的个数
		int[] topicCount = new int[K];
		for (int i = 0; i < wordTopics.length; i++) {
			topicCount[wordTopics[i]]++;
		}

		// 迭代更新
		for (int i = 0; i < iterations; i++) {
			for (int j = 0; j < words.size(); j++) {
				int oldTopic = wordTopics[j];
				topicCount[oldTopic]--;
				// 计算p(zi|z,w)
				float[] p = new float[K];
				for (int k = 0; k < K; k++) {
					p[k] = (topicCount[k] + alpha) / (words.size() + K * alpha) * phi[k][words.get(j)];
				}

				for (int k = 1; k < K; k++) {
					p[k] += p[k - 1];
				}
				double u = Math.random() * p[K - 1]; // p[] is unnormalised
				int newTopic;
				for (newTopic = 0; newTopic < K; newTopic++) {
					if (u < p[newTopic]) {
						break;
					}
				}
				topicCount[newTopic]++;
				wordTopics[j] = newTopic;
			}
		}

		for (int i = 0; i < topicVec.length; i++) {
			topicVec[i] = (topicCount[i] + alpha) / (words.size() + K * alpha);
		}
		
		return topicVec;
	}
}
