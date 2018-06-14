package com.ppp.dataminer.nlp.docclassify.vec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.doc2vec.util.Doc2Vec;
import com.ppp.dataminer.nlp.docclassify.tfidf.TFIDFClassifyApply;

/**
 * 数据生成工具
 * 
 * @author zhangwei
 *
 */
public class DataPrepare {
	private static DecimalFormat df = new DecimalFormat("######0.000");

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		// prepareDocVec();
		prepareTopicModelVec();
	}

	/**
	 * 生成主题模型向量分类器训练文件
	 * 
	 * @throws Exception
	 */
	private static void prepareTopicModelVec() throws Exception {
		// 文件写工具
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("model/topicModelVec.arff")));
		bw.write("@relation position");
		bw.newLine();
		for (int i = 0; i < 300; i++) {
			String line = "@attribute field" + i + " numeric";
			bw.write(line);
			bw.newLine();
		}
		bw.write(
				"@attribute position {翻译,销售行政及商务,服装_纺织_皮革,物流_仓储,公关_媒介,互联网_电子商务_网游,家政保洁,餐饮服务,物业管理,市场_营销,化工,金融_证券_期货_投资,质量安全,培训,保险,咨询_顾问,通信技术开发及应用,客服及支持,高级管理,银行,IT-品管、技术支持及其它,销售人员,机械机床,影视_媒体,人力资源,酒店旅游,汽车销售与服务,印刷包装,建筑工程与装潢,律师_法务_合规,IT-管理,医院_医疗_护理,计算机硬件,广告,公务员,汽车制造,百货零售,工程_机械_能源,运动健身,财务_审计_税务,行政_后勤,采购,房地产销售与中介,编辑出版,贸易,交通运输服务,环保,计算机软件,生产_营运,休闲娱乐,电子_电器_半导体_仪器仪表,科研,农_林_牧_渔,美容保健,房地产开发,教师,艺术_设计,生物_制药_医疗器械,网店淘宝,技工普工}");
		bw.newLine();
		bw.write("@data");
		bw.newLine();
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		String line = null;
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			int count = 0;
			while ((line = br.readLine()) != null && count++ < 500) {
				String[] segments = line.split("#&#&#");
				if (segments.length == 2) {
					double[] vec = TopicModelApply.convertContent2vec(segments[0] + segments[1]);
					String writeLine = "";
					for (double d : vec) {
						writeLine += df.format(d) + ",";
					}
					writeLine += sourceFile.getName();
					bw.write(writeLine);
					bw.newLine();
				}
			}
		}
		bw.close();
	}

	/**
	 * 生成docvector 文本向量，生成weka训练数据
	 * 
	 * @throws Exception
	 */
	private static void prepareDocVec() throws Exception {
		// 文件写工具
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("model/docVec.arff")));
		bw.write("@relation position");
		bw.newLine();
		for (int i = 0; i < 100; i++) {
			String line = "@attribute field" + i + " numeric";
			bw.write(line);
			bw.newLine();
		}
		bw.write(
				"@attribute position {翻译,销售行政及商务,服装_纺织_皮革,物流_仓储,公关_媒介,互联网_电子商务_网游,家政保洁,餐饮服务,物业管理,市场_营销,化工,金融_证券_期货_投资,质量安全,培训,保险,咨询_顾问,通信技术开发及应用,客服及支持,高级管理,银行,IT-品管、技术支持及其它,销售人员,机械机床,影视_媒体,人力资源,酒店旅游,汽车销售与服务,印刷包装,建筑工程与装潢,律师_法务_合规,IT-管理,医院_医疗_护理,计算机硬件,广告,公务员,汽车制造,百货零售,工程_机械_能源,运动健身,财务_审计_税务,行政_后勤,采购,房地产销售与中介,编辑出版,贸易,交通运输服务,环保,计算机软件,生产_营运,休闲娱乐,电子_电器_半导体_仪器仪表,科研,农_林_牧_渔,美容保健,房地产开发,教师,艺术_设计,生物_制药_医疗器械,网店淘宝,技工普工}");
		bw.newLine();
		bw.write("@data");
		bw.newLine();
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		String line = null;
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			int count = 0;
			while ((line = br.readLine()) != null && count++ < 500) {
				String[] segments = line.split("#&#&#");
				if (segments.length == 2) {
					float[] vec = genDocVec(segments[0], segments[1]);
					String writeLine = "";
					for (float d : vec) {
						writeLine += d + ",";
					}
					writeLine += sourceFile.getName();
					bw.write(writeLine);
					bw.newLine();
				}
			}
		}
		bw.close();
	}

	private static float[] genDocVec(String title, String content) {
		float[] vec = new float[100];
		List<Term> parse = ToAnalysis.parse(title + title + content);
		String[] words = new String[parse.size()];
		for (int i = 0; i < words.length; i++) {
			words[i] = parse.get(i).getName();
		}
		vec = Doc2Vec.getInstance().calcDocVec(words);
		return vec;
	}

	/*
	 * 生成tfidf支持度向量 生成weka训练数据
	 */
	public static void prerareTFIDFVec() throws Exception {
		Map<String, Integer> indexMap = getAttributeMap();
		// 文件写工具
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("model/keywordtfidfVec.arff")));
		bw.write("@relation position");
		bw.newLine();
		for (int i = 0; i < 60; i++) {
			String line = "@attribute field" + i + " numeric";
			bw.write(line);
			bw.newLine();
		}
		bw.write(
				"@attribute position {翻译,销售行政及商务,服装_纺织_皮革,物流_仓储,公关_媒介,互联网_电子商务_网游,家政保洁,餐饮服务,物业管理,市场_营销,化工,金融_证券_期货_投资,质量安全,培训,保险,咨询_顾问,通信技术开发及应用,客服及支持,高级管理,银行,IT-品管、技术支持及其它,销售人员,机械机床,影视_媒体,人力资源,酒店旅游,汽车销售与服务,印刷包装,建筑工程与装潢,律师_法务_合规,IT-管理,医院_医疗_护理,计算机硬件,广告,公务员,汽车制造,百货零售,工程_机械_能源,运动健身,财务_审计_税务,行政_后勤,采购,房地产销售与中介,编辑出版,贸易,交通运输服务,环保,计算机软件,生产_营运,休闲娱乐,电子_电器_半导体_仪器仪表,科研,农_林_牧_渔,美容保健,房地产开发,教师,艺术_设计,生物_制药_医疗器械,网店淘宝,技工普工}");
		bw.newLine();
		bw.write("@data");
		bw.newLine();
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		String line = null;
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			int count = 0;
			while ((line = br.readLine()) != null && count++ < 500) {
				String[] segments = line.split("#&#&#");
				if (segments.length == 2) {
					double[] vec = genTFIDFVec(segments[0], segments[1], indexMap);
					String writeLine = "";
					for (double d : vec) {
						writeLine += d + ",";
					}
					writeLine += sourceFile.getName();
					bw.write(writeLine);
					bw.newLine();
				}
			}
		}
		bw.close();
	}

	private static double[] genTFIDFVec(String title, String content, Map<String, Integer> attributeMap) {
		double[] vec = new double[60];
		List<WordPair> classifies = TFIDFClassifyApply.docClassify(title, content);
		double sum = 0;
		for (WordPair wp : classifies) {
			vec[attributeMap.get(wp.getWord())] = wp.getWeight();
			sum += wp.getWeight();
		}
		if (sum > 0) {
			for (int i = 0; i < vec.length; i++) {
				vec[i] = vec[i] / sum;
			}
		}
		return vec;
	}

	private static Map<String, Integer> getAttributeMap() throws Exception {
		Map<String, Integer> attributeMap = new HashMap<String, Integer>();
		BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(new File("model/name")), "utf8"));

		String line = null;
		while ((line = br.readLine()) != null) {
			String[] segments = line.split("-->");
			attributeMap.put(segments[0].replace("_", "/"), Integer.parseInt(segments[1].trim()));
		}
		br.close();
		return attributeMap;
	}

}
