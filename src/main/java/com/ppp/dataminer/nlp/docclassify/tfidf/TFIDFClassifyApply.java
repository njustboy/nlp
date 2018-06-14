package com.ppp.dataminer.nlp.docclassify.tfidf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.keyword.KeywordParserUtil;

/**
 * 基于TFIDF的文本分类应用
 * 
 * @author zhangwei
 *
 */
public class TFIDFClassifyApply {
	private static Map<String, Map<String, Float>> tfidfOfIndustry = new HashMap<String, Map<String, Float>>();
	// 词-->(分类，得分)
	public static Map<String, Map<String, Float>> tfidfMap = new HashMap<String, Map<String, Float>>();
	// 分类器分类-->(真实分类，个数)
	private static Map<String, Map<String, Float>> supportMap = new HashMap<String, Map<String, Float>>();

	private static Map<String,Float> categoryCount = new HashMap<String,Float>();
	
	private static Set<String> words = new HashSet<String>();
	
	// 返回分类个数
	private static int classifyNum = 3;
	// 修正得分时的步长
	private static float alpha = 0.2f;
	static {
		BufferedReader br = null;
		try {
			String rootPath = "";
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("/Users/zhimatech/workspace/cnn-text-classification/model/vocab.txt")), "UTF-8"));
			String line = null;
			while ((line = br.readLine()) != null) {
				words.add(line);
			}
			br.close();
			
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("model/tmptfidf")), "UTF-8"));
			line = null;
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("[\\{\\}]");
				if (segments.length == 2) {
//					if(!words.contains(segments[0].trim())){
//						continue;
//					}
					Map<String, Float> map = new HashMap<String, Float>();
					String[] wordPairs = segments[1].split(",");
					for (String wordPair : wordPairs) {
						String[] segs = wordPair.split("=");
						if (segs.length == 2) {
							map.put(segs[0].trim(), (float)Math.log(Float.valueOf(segs[1].trim())));
//							map.put(segs[0].trim(), Float.valueOf(segs[1].trim()));
						}
					}
					tfidfMap.put(segments[0].trim(), map);
				}
			}
			br.close();
			
			br = new BufferedReader(
					new InputStreamReader(new FileInputStream(new File(rootPath + "model/categoryCount.txt")), "UTF-8"));
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("\\s+");
				categoryCount.put(segments[0], (float)Math.log(Integer.parseInt(segments[1])));
			}
			br.close();

			br = new BufferedReader(
					new InputStreamReader(new FileInputStream(new File(rootPath + "model/supportMap")), "UTF-8"));
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("[\\{\\}]");
				if (segments.length == 2) {
					String[] wordPairs = segments[1].split(",");
					for (String wordPair : wordPairs) {
						String[] segs = wordPair.split("=");
						if (segs.length == 2) {
							Map<String, Float> map = null;
							if (supportMap.containsKey(segs[0].trim().replaceAll("_", "\\/"))) {
								map = supportMap.get(segs[0].trim().replaceAll("_", "\\/"));
							} else {
								map = new HashMap<String, Float>();
							}
							supportMap.put(segs[0].trim().replaceAll("_", "\\/"), map);
							map.put(segments[0].trim().replaceAll("_", "\\/"), Float.valueOf(segs[1].trim()));
						}
					}
				}
			}
		} catch (Exception e) {
		} finally {
			try {
				br.close();
			} catch (Exception ex) {

			}
		}
	}

	/**
	 * 分类接口
	 * 
	 * 分类的基本思想是将文本中不同词汇对分类的得分相加获得文本对分类的得分。
	 * 具体处理时通过加入词汇的语义权重/互信息以及在已知数据集上的分类错误分布情况进行分类得分调整
	 *
	 * 
	 * @param position
	 *            职位名称
	 * @param content
	 *            工作内容
	 * @return
	 */
	public static List<WordPair> docClassify(String position, String content) {
		List<WordPair> classifyList = new ArrayList<WordPair>();
		// 文本对不同分类的得分
		Map<String, Float> scoreMap = new HashMap<String, Float>();
		// 职位名称给予高权重

//		String line = position + position + content;
//		// 仅使用关键词
//		line = parseKeywords(line);
//		
//		List<Term> parse = ToAnalysis.parse(line);
		
//		String line = cleanString(content);
		String line = content;
		String[] parse = line.split("\\s+");

		/**
		 * 1 计算文本中不同单词本身的权重，在实际应用过程中发现加上单词权重后整体正确率下降比较严重，暂时不使用
		 */
		List<String> wordList = new ArrayList<String>();
		for (String term : parse) {
			if (tfidfMap.containsKey(term)) {
				wordList.add(term);
			}
		}
		// // 词权重
		// Map<String, Float> weightMap = TFIDFUtil.getSemanticScore(wordList);
		// for (Entry<String, Float> entry : weightMap.entrySet()) {
		// // 词的权重=词的语义得分*词的分类离散度
		// weightMap.put(entry.getKey(), entry.getValue() *
		// TFIDFUtil.calcCV(tfidfMap.get(entry.getKey())));
		// }

		/**
		 * 2 计算每个词的最大支持分类
		 */
		// 词-->最大支持度分类
		Map<String, String> wordClassify = new HashMap<String, String>();
		for (String word : wordList) {
			Float maxScore = 0f;
			for (String key : tfidfMap.get(word).keySet()) {
				if (tfidfMap.get(word).get(key) > maxScore) {
					maxScore = tfidfMap.get(word).get(key);
					wordClassify.put(word, key);
				}
			}
		}

		/**
		 * 3 计算文本的整体分类得分
		 */
//		for (Term term : parse) {
		for(String term:parse){
			if (tfidfMap.containsKey(term)) {
				Map<String, Float> tmpMap = new HashMap<String, Float>();
				for (String key : tfidfMap.get(term).keySet()) {
					tmpMap.put(key, tfidfMap.get(term).get(key));
					// tmpMap.put(key, tfidfMap.get(term.getName()).get(key) *
					// weightMap.get(term.getName()));
				}
				TFIDFUtil.mapAdd(scoreMap, tmpMap);
			}
		}

		for(String category:scoreMap.keySet()){
//			scoreMap.put(category, scoreMap.get(category)+categoryCount.get(category));
			scoreMap.put(category, scoreMap.get(category));
		}
		
		List<WordPair> tmpList = new ArrayList<WordPair>();
		for (String name : scoreMap.keySet()) {
			tmpList.add(new WordPair(name.replaceAll("_", "\\/"), scoreMap.get(name)));
		}
		Collections.sort(tmpList);
		int maxIndex = tmpList.size() > classifyNum ? classifyNum : tmpList.size();
		for (int i = 0; i < maxIndex; i++) {
			classifyList.add(tmpList.get(i));
		}

		if (classifyList.size() <= 1) {
			return classifyList;
		}
		

//		/**
//		 * 4 基于互信息的思想修正分类得分
//		 */
//		float one2two = 1f;
//		float two2one = 1f;
//		int classify1Count = 0;
//		int classify2Count = 0;
//		for (String word : wordList) {
//			Map<String, Float> tmpMap = tfidfMap.get(word);
//
//			if (wordClassify.get(word).equals(classifyList.get(0).getWord())) {
//				float classify2Score = tmpMap.containsKey(classifyList.get(1).getWord())
//						? tmpMap.get(classifyList.get(1).getWord()) : 0.01f;
//				one2two += tmpMap.get(classifyList.get(0).getWord()) / classify2Score;
//				classify1Count++;
//			}
//
//			if (wordClassify.get(word).equals(classifyList.get(1).getWord())) {
//				float classify1Score = tmpMap.containsKey(classifyList.get(0).getWord())
//						? tmpMap.get(classifyList.get(0).getWord()) : 0.01f;
//				two2one += tmpMap.get(classifyList.get(1).getWord()) / classify1Score;
//				classify2Count++;
//			}
//
//		}
//		if (classify1Count > 0) {
//			one2two /= classify1Count;
//		}
//		if (classify2Count > 0) {
//			two2one /= classify2Count;
//		}
//
//		if (one2two < two2one) {
//			float ff = (float) Math.log(two2one / one2two) + 1;
//			WordPair firstClassify = classifyList.get(0);
//			for (String word : wordList) {
//				Map<String, Float> tmpMap = tfidfMap.get(word);
//				if (tmpMap.containsKey(firstClassify.getWord())) {
//					float score = tmpMap.get(firstClassify.getWord());
//					score = (1 / ff - 1) * score;
//					classifyList.get(0).setWeight(classifyList.get(0).getWeight() + score * alpha);
//					classifyList.get(1).setWeight(classifyList.get(1).getWeight() - score * alpha);
//				}
//			}
//		}
//
//		if (classifyList.size() >= 3) {
//			float one2three = 1f;
//			float three2one = 1f;
//			classify1Count = 0;
//			int classify3Count = 0;
//			for (String word : wordList) {
//				Map<String, Float> tmpMap = tfidfMap.get(word);
//				if (wordClassify.get(word).equals(classifyList.get(0).getWord())) {
//					float classify3Score = tmpMap.containsKey(classifyList.get(2).getWord())
//							? tmpMap.get(classifyList.get(2).getWord()) : 0.01f;
//					one2three += tmpMap.get(classifyList.get(0).getWord()) / classify3Score;
//					classify1Count++;
//				}
//
//				if (wordClassify.get(word).equals(classifyList.get(2).getWord())) {
//					float classify1Score = tmpMap.containsKey(classifyList.get(0).getWord())
//							? tmpMap.get(classifyList.get(0).getWord()) : 0.01f;
//					three2one += tmpMap.get(classifyList.get(2).getWord()) / classify1Score;
//					classify3Count++;
//				}
//			}
//
//			if (classify1Count > 0) {
//				one2three /= classify1Count;
//			}
//			if (classify3Count > 0) {
//				three2one /= classify3Count;
//			}
//
//			if (one2three < three2one) {
//				float ff = (float) Math.log(three2one / one2three) + 1;
//				WordPair firstClassify = classifyList.get(0);
//				for (String word : wordList) {
//					Map<String, Float> tmpMap = tfidfMap.get(word);
//					if (tmpMap.containsKey(firstClassify.getWord())) {
//						float score = tmpMap.get(firstClassify.getWord());
//						score = (1 / ff - 1) * score;
//						classifyList.get(0).setWeight(classifyList.get(0).getWeight() + score * alpha);
//						classifyList.get(2).setWeight(classifyList.get(2).getWeight() - score * alpha);
//					}
//				}
//			}
//		}
//
//		Collections.sort(classifyList);

		/**
		 * 5 基于已知数据集的分类情况修正得分
		 */
//		reSort(classifyList);

		return classifyList;
	}

	/**
	 * 行业分类接口
	 * 
	 * @param company
	 *            公司名称
	 * @param content
	 *            工作内容
	 * @return
	 */
	public static List<WordPair> docClassifyForIndustry(String industry, String company, String content) {
		List<WordPair> classifyList = new ArrayList<WordPair>();

		Map<String, Float> scoreMap = new HashMap<String, Float>();
		List<Term> parse = ToAnalysis.parse(company + industry + content);
		for (Term term : parse) {
			if (tfidfOfIndustry.containsKey(term.getName())) {
				TFIDFUtil.mapAdd(scoreMap, tfidfOfIndustry.get(term.getName()));
			}
		}

		List<WordPair> tmpList = new ArrayList<WordPair>();
		for (String name : scoreMap.keySet()) {
			tmpList.add(new WordPair(name.replaceAll("_", "\\/"), scoreMap.get(name)));
		}
		Collections.sort(tmpList);

		int maxIndex = tmpList.size() > classifyNum ? classifyNum : tmpList.size();
		for (int i = 0; i < maxIndex; i++) {
			classifyList.add(tmpList.get(i));
		}

		return classifyList;
	}

	/**
	 * 
	 * 重排序的基本思想是：如果真实分类B有可能被错分为A，那么当分类器得到的结果为A时应该适当减小A的得分，增加B的得分
	 * 
	 * @param classifyList
	 */
	private static void reSort(List<WordPair> classifyList) {
		Map<String, Float> deltaMap = new HashMap<String, Float>();
		for (WordPair wp : classifyList) {
			deltaMap.put(wp.getWord(), 0f);
		}
		for (WordPair wp : classifyList) {
			float sum = 0f;
			// 统计分类器分出的某个分类的总数
			for (Entry<String, Float> entry : supportMap.get(wp.getWord()).entrySet()) {
				sum += entry.getValue();
			}

			for (WordPair wp1 : classifyList) {
				if (wp.getWord().equals(wp1.getWord())) {
					continue;
				}
				if (supportMap.get(wp.getWord()).containsKey(wp1.getWord())) {
					deltaMap.put(wp.getWord(), deltaMap.get(wp.getWord())
							- 2 * (float) wp.getWeight() * supportMap.get(wp.getWord()).get(wp1.getWord()) / sum);
					deltaMap.put(wp1.getWord(), deltaMap.get(wp1.getWord())
							+ 2 * (float) wp.getWeight() * supportMap.get(wp.getWord()).get(wp1.getWord()) / sum);
				}
			}
		}

		for (WordPair wp : classifyList) {
			wp.setWeight(wp.getWeight() + deltaMap.get(wp.getWord()));
		}

		Collections.sort(classifyList);
	}

	private static String parseKeywords(String line) {
		String keywords = "";
		List<WordPair> parseKeywords = KeywordParserUtil.parseKeywords(line, true);
		for (WordPair wp : parseKeywords) {
			keywords += wp.getWord() + " ";
		}
		return keywords;
	}
	
	 private static String cleanString(String str){
		   str = str.replaceAll("[^A-Za-z0-9(),!?\'\\`]", " ");
		   str = str.replaceAll("\\'s", " \\'s");
		   str = str.replaceAll("\\'ve", " \\'ve");
		   str = str.replaceAll("n\\'t", " n\\'t");
		   str = str.replaceAll("\\'re", " \\'re");
		   str = str.replaceAll("\\'d", " \\'d");
		   str = str.replaceAll("\\'ll", " \\'ll");
		   str = str.replaceAll(",", " , ");
		   str = str.replaceAll("!", " ! ");
		   str = str.replaceAll("\\(", " \\( ");
		   str = str.replaceAll("\\)", " \\) ");
		   str = str.replaceAll("\\?", " \\? ");
		   str = str.replaceAll("\\s{2,}", " ");
		   return str;
	   }
}
