package com.ppp.dataminer.nlp.keyword;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.topicmodel.data.ScoreComparable;
import com.ppp.dataminer.nlp.topicmodel.plsa.PLSAInference;

/**
 * 通用的关键词提取工具类
 * 
 * @author zhimatech
 *
 */
public class KeywordParserUtil {
	/**
	 * 静态生成词性并赋予不同词性不同重要性
	 */
	public static Map<String, Double> POS_SCORE = new HashMap<String, Double>();
	static {
		POS_SCORE.put("null", 0.0);
		POS_SCORE.put("w", 0.0);
		POS_SCORE.put("en", 5.0);
		POS_SCORE.put("num", 0.0);
		POS_SCORE.put("nr", 0.0);
		POS_SCORE.put("n", 2.5);
		POS_SCORE.put("nrf", 2.5);
		POS_SCORE.put("nw", 2.5);
		POS_SCORE.put("nt", 2.5);
		POS_SCORE.put("l", 1.0);
		POS_SCORE.put("a", 0.5);
		POS_SCORE.put("nz", 1.0);
		POS_SCORE.put("v", 1.0);
		POS_SCORE.put("vn", 1.5);
		// 自定义词性应该给与高权重
		POS_SCORE.put("userDefine", 5.0);
		initStopwords();
	}

	public static List<String> simpleParseKeywords(String content, int count) {
		List<String> keywords = new ArrayList<String>();
		// 主题分布
		float[] topicPros = PLSAInference.getInstance().plsaInference(content);
		// 主题对应的索引
		Integer[] index = new Integer[topicPros.length];
		for (int i = 0; i < index.length; i++) {
			index[i] = i;
		}
		Arrays.sort(index, new ScoreComparable(topicPros));

		List<Term> terms = ToAnalysis.parse(content);
		Map<Integer, List<String>> topicKeywords = PLSAInference.getInstance().getPlsa().getTopicKeywords();
		for (int i = 0; i < index.length; i++) {
			List<String> list = topicKeywords.get(index[i]);
			for (Term term : terms) {
				if (list.contains(term.getName()) && !keywords.contains(term.getName())) {
					keywords.add(term.getName());
				}
				if (keywords.size() >= count) {
					break;
				}
			}
			if (keywords.size() >= count) {
				break;
			}
		}

		return keywords;
	}

	/**
	 * 
	 * @param content
	 *            输入文本
	 * @param count
	 *            需要提取的关键词数量
	 * @return 带权值的关键词列表
	 */
	public static List<WordPair> parseKeywords(String content, int count) {
		List<WordPair> keywords = new ArrayList<WordPair>();

		// 分词
		content = content.toLowerCase();
		List<Term> wordlist = ToAnalysis.parse(content);
		// 去停用词
		wordlist = FilterModifWord.modifResult(wordlist);
		// 文本中包含的词（去重）
		Set<String> wordSet = new HashSet<String>();
		// 记录每个词出现的次数
		Map<String, Double> tfMap = new HashMap<String, Double>();
		// 词性得分
		Map<String, Double> cixingWeightMap = new HashMap<String, Double>();
		for (Term term : wordlist) {
			// 关键词不能是单个的字
			if (term.getName() == null || term.getName().length() < 2) {
				continue;
			}
			// 记录出现的词
			wordSet.add(term.getName());
			// 计算关键字TF
			if (tfMap.containsKey(term.getName())) {
				tfMap.put(term.getName(), tfMap.get(term.getName()) + 1);
			} else {
				tfMap.put(term.getName(), 1.0);
			}

			// 计算词性关键字及权重
			double weightcixing = getweightcixing(term, content.length());
			if (weightcixing == 0) {
				continue;
			}
			// 记录第一次出现的得分
			if (!cixingWeightMap.containsKey(term.getName())) {
				cixingWeightMap.put(term.getName(), weightcixing);
			}
		}

		// 每个词的主题支持度
		Map<String, Double> semanticMap = new HashMap<String, Double>();
		// 计算主题支持度
		for (String word : wordSet) {
			// 主题支持度得分
			double semanticScore = SimilarWordUtil.getSemanticScore(word, wordSet);
			semanticMap.put(word, semanticScore);
		}

		tfMap = normalized(tfMap);
		cixingWeightMap = normalized(cixingWeightMap);
		semanticMap = normalized(semanticMap);

		List<WordPair> allWords = update(tfMap, cixingWeightMap, semanticMap);
		Collections.sort(allWords);

		if (allWords.size() > Config.KEYWORD_MAXCOUNT) {
			for (int i = 0; i < Config.KEYWORD_MAXCOUNT; i++) {
				keywords.add(allWords.get(i));
			}
			return keywords;
		} else {
			return allWords;
		}
	}

	/**
	 * 
	 * @param term
	 * @param length
	 * @return 由词性生成关键词的词性权重
	 */
	private static double getweightcixing(Term term, int length) {
		if (term.getName().trim().length() < 2) {
			return 0;
		}
		String pos = term.natrue().natureStr;
		Double posScore = POS_SCORE.get(pos);

		if (posScore == null) {
			posScore = 1.0;
		} else if (posScore == 0) {
			return 0;
		}
		return (length - term.getOffe()) * posScore / (double) length;
	}

	/**
	 * 
	 * @return 初始化停用此表（包括停用词与停用词性）
	 */
	private static boolean initStopwords() {
		File stopWordsFile = new File("library/stopword.dic");
		String stopWord = null;
		FileInputStream fis = null;
		InputStreamReader isr = null;
		BufferedReader br = null;
		try {
			fis = new FileInputStream(stopWordsFile);
			isr = new InputStreamReader(fis, "utf-8");
			br = new BufferedReader(isr);
			stopWord = br.readLine();
			while (stopWord != null) {
				FilterModifWord.insertStopWord(stopWord);
				stopWord = br.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		} finally {
			try {
				br.close();
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
		return true;
	}

	/**
	 * 
	 * @param tempMap
	 * @return 归一化value值，消除组间误差
	 */
	private static Map<String, Double> normalized(Map<String, Double> tempMap) {
		Set<String> keyset = tempMap.keySet();
		double maxvalue = 0.0;
		double minvalue = Double.MAX_VALUE;
		for (String word : keyset) {

			maxvalue = tempMap.get(word) > maxvalue ? tempMap.get(word) : maxvalue;
			minvalue = tempMap.get(word) < minvalue ? tempMap.get(word) : minvalue;
		}
		double dvalue = maxvalue - minvalue;
		double truevalue = 0.0;
		if (dvalue > 0) {
			for (String wd : keyset) {
				truevalue = (tempMap.get(wd) - minvalue) / dvalue;
				tempMap.put(wd, truevalue);
			}
		} else {
			for (String wd : keyset) {
				tempMap.put(wd, 0.5);
			}
		}
		return tempMap;
	}

	/**
	 * 
	 * @param tempMap
	 *            词频得分表
	 * @param tempcixing
	 *            词性得分表
	 * @param tmpSemanticMap
	 *            语义得分表
	 * @param tmpTopicMap
	 *            主题得分表
	 * @return
	 */
	private static List<WordPair> update(Map<String, Double> tfMap, Map<String, Double> cixingWeightMap,
			Map<String, Double> semanticMap) {
		List<WordPair> returnlist = new ArrayList<WordPair>();

		Set<String> keyset = tfMap.keySet();
		double truevalue = 0.0;
		for (String ky : keyset) {
			Double mapvalue = tfMap.get(ky);
			Double cixingvalue = cixingWeightMap.get(ky);
			Double semanticValue = semanticMap.get(ky);
			// 处理一方不包含对应关键字的情况，value值域为（0，1）之间，赋值应该在此区间
			if (cixingvalue == null || cixingvalue.isNaN()) {
				cixingvalue = Config.UPDATE_WHEN_NULL;
			}
			if (semanticValue == null || semanticValue.isNaN()) {
				semanticValue = Config.UPDATE_WHEN_NULL;
			}

			truevalue = Config.TFIDF_PERCENT * mapvalue + Config.CIXING_PERCENT * cixingvalue
					+ (1 - Config.TFIDF_PERCENT - Config.CIXING_PERCENT) * semanticValue;

			WordPair wp = new WordPair();
			wp.setWeight(truevalue);
			wp.setWord(ky);
			returnlist.add(wp);
		}

		return returnlist;
	}

}
