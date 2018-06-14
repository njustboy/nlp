package com.ppp.dataminer.nlp.docclassify.test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.docclassify.tfidf.TFIDFClassifyApply;

/**
 * 批量测试类 用于测试不同分类器在标准测试集上表现
 * 
 * 只用于本地测试
 * 
 * @author zhangwei
 *
 */
public class ClassifyBatchTest {
	public static void main(String[] args) throws Exception {
		Map<String, List<String>> cnnClassifyMap = getCNNClassifyResult();

		// 实际分类-->(分类器分类，个数)
		Map<String, Map<String, Double>> map = new HashMap<String, Map<String, Double>>();
		Map<Integer,List<Double>> scoreMap = new HashMap<Integer,List<Double>>();
		dirTest("testwords", true, map, cnnClassifyMap,scoreMap);
		
		double sum0 = 0;
		for(double d:scoreMap.get(0)){
			sum0 += d;
		}
		double sum1 = 0;
		for(double d:scoreMap.get(1)){
			sum1 += d;
		}
		System.out.println("分类正确样本第一分类与第二分类权重比为:"+sum1/scoreMap.get(1).size());
		System.out.println("分类错误样本第一分类与第二分类权重比为:"+sum0/scoreMap.get(0).size());
		
		// IOUtil.writeMap(map, "model/supportMap", "utf-8");
		// 分类--真实分类个数
		Map<String, Double> realCategoryCount = new HashMap<String, Double>();
		// 分类--分类器个数
		Map<String, Double> classifyCategoryCount = new HashMap<String, Double>();
		// 分类--分类正确个数
		Map<String, Double> realClassifyCount = new HashMap<String, Double>();
		// 真实分类
		for (String realCategory : map.keySet()) {
			// 分类器分类--个数
			for (Entry<String, Double> entry : map.get(realCategory).entrySet()) {
				if (realCategoryCount.containsKey(realCategory)) {
					realCategoryCount.put(realCategory, realCategoryCount.get(realCategory) + entry.getValue());
				} else {
					realCategoryCount.put(realCategory, entry.getValue());
				}

				if (realCategory.equals(entry.getKey())) {
					realClassifyCount.put(realCategory, entry.getValue());
				}

				if (classifyCategoryCount.containsKey(entry.getKey())) {
					classifyCategoryCount.put(entry.getKey(),
							classifyCategoryCount.get(entry.getKey()) + entry.getValue());
				} else {
					classifyCategoryCount.put(entry.getKey(), entry.getValue());
				}

			}
		}

		double recallAvg = 0;
		double correctAvg = 0;
		double recallF = 0;
		double correctF = 0;
		double totalCount = 0;
		double correctCount = 0;
		for (String classify : realClassifyCount.keySet()) {
			totalCount += realCategoryCount.get(classify);
			correctCount += realClassifyCount.get(classify);
			recallAvg += realClassifyCount.get(classify) / realCategoryCount.get(classify);
			correctAvg += realClassifyCount.get(classify) / classifyCategoryCount.get(classify);
			System.out.println(classify + "共有真实样本" + realCategoryCount.get(classify) + "条，分类样本"
					+ classifyCategoryCount.get(classify) + "条，分类正确" + realClassifyCount.get(classify) + "条。召回率为"
					+ realClassifyCount.get(classify) / realCategoryCount.get(classify) + " 准确率为"
					+ realClassifyCount.get(classify) / classifyCategoryCount.get(classify));
		}
		recallAvg = recallAvg / 60;
		correctAvg = correctAvg / 60;

		for (String classify : realClassifyCount.keySet()) {
			recallF += Math.pow(realClassifyCount.get(classify) / realCategoryCount.get(classify) - recallAvg, 2);
			correctF += Math.pow(realClassifyCount.get(classify) / classifyCategoryCount.get(classify) - correctAvg, 2);
		}
		recallF = Math.sqrt(recallF / 2);
		correctF = Math.sqrt(correctF / 2);
		System.out.println("共有样本：" + totalCount + ",整体正确率为：" + correctCount / totalCount + "正确率平均值为：" + correctAvg
				+ "正确率标准差为：" + correctF + ",召回率平均值为：" + recallAvg + "召回率标准差为：" + recallF);
	}

	/**
	 * 多文本批量测试
	 * 
	 * @param modeFile
	 * @param corpusFile
	 * @param beStrict
	 *            是否严格匹配
	 * @throws Exception
	 */
	public static void dirTest(String corpusFile, boolean beStrict, Map<String, Map<String, Double>> map,
			Map<String, List<String>> cnnClassifyMap,Map<Integer,List<Double>> scoreMap) throws Exception {
		int totalCount = 0;
		int correctCount = 0;
		File dirFile = new File(corpusFile);
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("model/falseresult.txt")));
		for (File listFile : dirFile.listFiles()) {
			String result = fileTest(listFile, beStrict, map, bw, cnnClassifyMap,scoreMap);
			String[] segments = result.split("-");
			totalCount += Integer.parseInt(segments[0]);
			correctCount += Integer.parseInt(segments[1]);
		}
		bw.flush();
		bw.close();
		System.out.println(
				"共有样本" + totalCount + "条，分类正确样本" + correctCount + "条，分类正确率为" + (double) correctCount / totalCount);
	}

	private static String fileTest(File corpusFile, boolean beStrict, Map<String, Map<String, Double>> map,
			BufferedWriter bw, Map<String, List<String>> cnnClassifyMap,Map<Integer,List<Double>> scoreMap) {
		String realClassify = corpusFile.getName();
		int count = 0;
		int correctCount = 0;
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(corpusFile));
			String line = "";
			Map<String, Double> value = new HashMap<String, Double>();
			map.put(corpusFile.getName().replaceAll("_", "/"), value);
			while ((line = br.readLine()) != null) {
				count++;
				String position = "";
				String content = "";
				String[] segments = line.split("#&#&#");
				if (segments.length == 2) {
					position = segments[0];
					content = segments[1];
				} else {
					content = line;
				}
				// 基于doc2vec的文本分类
				// List<WordPair> docClassify =
				// DocVecClassifyApply.docClassify(position, content);
				// 基于TFIDF的文本分类
				List<WordPair> docClassify = TFIDFClassifyApply.docClassify("", content);
				// List<WordPair> docClassify =
				// DocVecClassifyApply.docClassify(position, content);
				// List<WordPair> docClassify =
				// TopicModelVecClassifyApply.docClassify(position, content);
				// List<WordPair> docClassify =
				// TFIDFVecClassifyApply.docClassify(position, content);

				// List<WordPair> docClassify =
				// SMOClassifier.classifyInput(position, content);

				if (docClassify.size() == 0) {
					continue;
				}

				if (value.containsKey(docClassify.get(0).getWord())) {
					value.put(docClassify.get(0).getWord(), value.get(docClassify.get(0).getWord()) + 1);
				} else {
					value.put(docClassify.get(0).getWord(), 1.0);
				}
				
				if (!docClassify.get(0).getWord().equals(realClassify.replaceAll("_", "\\/"))&&docClassify.size()>1) {
					// 第一分类错误
					if(scoreMap.containsKey(0)){
						scoreMap.get(0).add(docClassify.get(0).getWeight()/docClassify.get(1).getWeight());
					}else{
						List<Double> list = new ArrayList<Double>();
						list.add(docClassify.get(0).getWeight()/docClassify.get(1).getWeight());
						scoreMap.put(0, list);
					}
				}else if(docClassify.size()>1){
					// 第一分类正确
					if(scoreMap.containsKey(1)){
						scoreMap.get(1).add(docClassify.get(0).getWeight()/docClassify.get(1).getWeight());
					}else{
						List<Double> list = new ArrayList<Double>();
						list.add(docClassify.get(0).getWeight()/docClassify.get(1).getWeight());
						scoreMap.put(1, list);
					}
				}
				
//				if(cnnClassifyMap.containsKey(realClassify)){
//				if (cnnClassifyMap.get(realClassify).get(count-1).equals(realClassify)) {
//					correctCount++;
//				}
//				}
				
				if (beStrict) {
					if (docClassify.size() > 1 && docClassify.get(0).getWeight() < 1.1 * docClassify.get(1).getWeight()
							&& cnnClassifyMap.containsKey(realClassify)) {
						if (cnnClassifyMap.get(realClassify).get(count-1).equals(realClassify)) {
							correctCount++;
						}
					} else {
						if (docClassify.get(0).getWord().equals(realClassify.replaceAll("_", "\\/"))) {
							correctCount++;
						} else {
							bw.write(realClassify + "-->" + docClassify + "-->" + line);
							bw.newLine();
						}
					}
				} else {
					for (WordPair wp : docClassify) {
						if (wp.getWord().equals(realClassify.replaceAll("_", "\\/"))) {
							correctCount++;
							break;
						}
					}
				}

			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (Exception ex) {

			}
		}
		System.out.println(
				realClassify + "共有" + count + "条样本，分类正确样本" + correctCount + "条，召回率为" + (double) correctCount / count);
		return count + "-" + correctCount;
	}

	private static Map<String, List<String>> getCNNClassifyResult() throws Exception {
		Map<String, List<String>> classifyResultMap = new HashMap<String, List<String>>();
		File file = new File("/Users/zhimatech/workspace/cnn-text-classification/runs/r_60_200_6/result_classify");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] segments = line.split(":");
			String[] classifySegments = segments[1].split("-->");
			if(classifyResultMap.containsKey(classifySegments[0])){
				classifyResultMap.get(classifySegments[0]).add(classifySegments[1]);
			}else{
				List<String> list = new ArrayList<String>();
				list.add(classifySegments[1]);
				classifyResultMap.put(classifySegments[0], list);
			}
		}
		br.close();
		return classifyResultMap;
	}
}
