package com.ppp.dataminer.nlp.topicmodel.plsa;

import java.util.ArrayList;
import java.util.List;

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

	/**
	 * 模型初始化
	 * @param modelPath
	 */
	public void initializeModel(String modelPath){
		FileUtil.readLines(modelPath+"wordDic", wordDic);
		M = wordDic.size();
		topicTermPros = FileUtil.read2DArray(modelPath+"model_fast_100.topicTermPros");
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
