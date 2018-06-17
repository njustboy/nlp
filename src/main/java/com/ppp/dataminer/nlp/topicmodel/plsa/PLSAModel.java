package com.ppp.dataminer.nlp.topicmodel.plsa;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ppp.dataminer.nlp.topicmodel.util.FileUtil;

/**
 * PLSA模型，在应用是使用
 * 
 * @author zhimatech
 *
 */
public class PLSAModel {
	// 单词数量，词向量长度
	private int M;
	// 主题--词 分布
	private float[][] topicTermPros;// p(w|z)
	// word list
	private List<String> wordDic = new ArrayList<String>();

	private Map<Integer, List<String>> topicKeywords = new HashMap<Integer, List<String>>();

	/**
	 * 模型初始化
	 * 
	 * @param modelPath
	 */
	public void initializeModel(String modelPath) {
		FileUtil.readLines(modelPath + "wordDic", wordDic);
		M = wordDic.size();
		topicTermPros = FileUtil.read2DArray(modelPath + "model_200.topicTermPros");

		List<String> lines = new ArrayList<String>();
		FileUtil.readLines(new File(modelPath + "model_200.zterms"), lines);
		String[] segments = null;
		for (int i = 0; i < lines.size(); i++) {
			segments = lines.get(i).split("\\s+");
			List<String> keywords = new ArrayList<String>();
			for (int j = 1; j < segments.length; j++) {
				keywords.add(segments[j]);
			}
			topicKeywords.put(i, keywords);
		}
	}

	public int getM() {
		return M;
	}

	public void setM(int m) {
		M = m;
	}

	public float[][] getTopicTermPros() {
		return topicTermPros;
	}

	public Map<Integer, List<String>> getTopicKeywords() {
		return topicKeywords;
	}

	public void setTopicKeywords(Map<Integer, List<String>> topicKeywords) {
		this.topicKeywords = topicKeywords;
	}

	public void setTopicTermPros(float[][] topicTermPros) {
		this.topicTermPros = topicTermPros;
	}

	public List<String> getWordDic() {
		return wordDic;
	}

	public void setWordDic(List<String> wordDic) {
		this.wordDic = wordDic;
	}
}
