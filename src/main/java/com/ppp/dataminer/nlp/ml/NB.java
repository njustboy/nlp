package com.ppp.dataminer.nlp.ml;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 朴素贝叶斯的简单实现
 * 
 * @author zhimatech
 *
 */
public class NB {
	private boolean[] isNumFeature;
	// 数值型特征的均值
	private Map<String, Map<Integer, Double>> avgMap = new HashMap<String, Map<Integer, Double>>();
	// 数值型特征的方差
	private Map<String, Map<Integer, Double>> varMap = new HashMap<String, Map<Integer, Double>>();
	// 离散型特征的统计个数
	// 特征id--分类--特征值--个数
	private Map<Integer, Map<String, Map<String, Integer>>> countMap = new HashMap<Integer, Map<String, Map<String, Integer>>>();
	// 分类个数
	private Map<String, Integer> classifyCountMap = new HashMap<String, Integer>();

	public void train(List<String[]> trainDatas) {
		initFeatureCharacter();

		for (String[] data : trainDatas) {
			if (classifyCountMap.containsKey(data[data.length - 1])) {
				classifyCountMap.put(data[data.length - 1], classifyCountMap.get(data[data.length - 1]) + 1);
			} else {
				classifyCountMap.put(data[data.length - 1], 1);
			}
			// 分类标志
			for (int i = 0; i < data.length - 1; i++) {
				if (isNumFeature[i]) {
					Map<Integer, Double> subAvgMap = null;
					if (avgMap.containsKey(data[data.length - 1])) {
						subAvgMap = avgMap.get(data[data.length - 1]);
					} else {
						subAvgMap = new HashMap<Integer, Double>();
						avgMap.put(data[data.length - 1], subAvgMap);
					}
					// 对于数值型特征统计均值
					if (subAvgMap.containsKey(i)) {
						subAvgMap.put(i, subAvgMap.get(i) + Double.parseDouble(data[i]));
					} else {
						subAvgMap.put(i, Double.parseDouble(data[i]));
					}
				} else {
					if (countMap.containsKey(i)) {
						Map<String, Map<String, Integer>> subCountMap = countMap.get(i);
						if (subCountMap.containsKey(data[data.length - 1])) {
							Map<String, Integer> featureMap = subCountMap.get(data[data.length - 1]);
							if (featureMap.containsKey(data[i])) {
								featureMap.put(data[i], featureMap.get(data[i]) + 1);
							} else {
								featureMap.put(data[i], 1);
							}
						} else {
							Map<String, Integer> featureMap = new HashMap<String, Integer>();
							featureMap.put(data[i], 1);
							subCountMap.put(data[data.length - 1], featureMap);
						}
					} else {
						Map<String, Map<String, Integer>> subCountMap = new HashMap<String, Map<String, Integer>>();
						Map<String, Integer> featureMap = new HashMap<String, Integer>();
						featureMap.put(data[i], 1);
						subCountMap.put(data[data.length - 1], featureMap);
						countMap.put(i, subCountMap);
					}
				}
			}
		}

		for (String classify : avgMap.keySet()) {
			Map<Integer, Double> subMap = avgMap.get(classify);
			for (Integer key : subMap.keySet()) {
				// 计算均值
				subMap.put(key, subMap.get(key) / classifyCountMap.get(classify));
			}
		}

		// 计算方差
		for (String[] data : trainDatas) {
			Map<Integer, Double> subVarMap = null;
			if (varMap.containsKey(data[data.length - 1])) {
				subVarMap = varMap.get(data[data.length - 1]);
			} else {
				subVarMap = new HashMap<Integer, Double>();
				varMap.put(data[data.length - 1], subVarMap);
			}

			Map<Integer, Double> subAvgMap = avgMap.get(data[data.length - 1]);

			for (Integer key : subAvgMap.keySet()) {
				if (subVarMap.containsKey(key)) {
					subVarMap.put(key,
							subVarMap.get(key) + Math.pow(subAvgMap.get(key) - Double.parseDouble(data[key]), 2));

				} else {
					subVarMap.put(key, Math.pow(subAvgMap.get(key) - Double.parseDouble(data[key]), 2));
				}
			}
		}

		for (String classify : varMap.keySet()) {
			Map<Integer, Double> subMap = varMap.get(classify);
			for (Integer key : subMap.keySet()) {
				// 计算均值
				subMap.put(key, subMap.get(key) / classifyCountMap.get(classify));
			}
		}

	}

	public void test(List<String[]> testDatas) {
		int correctCount = 0;
		for (String[] data : testDatas) {
			Map<String, Double> scoreMap = new HashMap<String, Double>();
			for (String classify : classifyCountMap.keySet()) {
				for (int i = 0; i < data.length - 1; i++) {
					if (isNumFeature[i]) {
						// 如果是数值型，则使用高斯公式算概率
						double thisScore = gauss(avgMap.get(classify).get(i), varMap.get(classify).get(i),
								Double.parseDouble(data[i]));
						if (scoreMap.containsKey(classify)) {
							scoreMap.put(classify, scoreMap.get(classify) * thisScore);
						} else {
							scoreMap.put(classify, thisScore);
						}
					} else {
						Map<String, Integer> subCountMap = countMap.get(i).get(classify);
						double thisScore = 0;
						if (subCountMap.containsKey(data[i])) {
							thisScore = 1.0 * (subCountMap.get(data[i]) + 1.0) / (classifyCountMap.get(classify) + 1);
						} else {
							thisScore = 1.0 / (classifyCountMap.get(classify) + 1);
						}
						if (scoreMap.containsKey(classify)) {
							scoreMap.put(classify, scoreMap.get(classify) * thisScore);
						} else {
							scoreMap.put(classify, thisScore);
						}
					}
				}
			}
			for (String classify : scoreMap.keySet()) {
				scoreMap.put(classify, scoreMap.get(classify) * classifyCountMap.get(classify));
			}

			String thisClassify = "";
			double maxScore = 0;
			for (String classify : scoreMap.keySet()) {
				if (scoreMap.get(classify) > maxScore) {
					maxScore = scoreMap.get(classify);
					thisClassify = classify;
				}
			}
			if (thisClassify.equals(data[data.length - 1])) {
				correctCount++;
			}
		}

		System.out.println(testDatas.size());
		System.out.println(correctCount);
	}

	private double gauss(double avg, double var, double d) {
		return 1 / Math.pow(2 * 3.14 * var, 0.5) * Math.pow(Math.E, -Math.pow(avg - d, 2) / (2 * var));
	}

	private void initFeatureCharacter() {
		isNumFeature = new boolean[] { true, false, true, false, true, false, false, false, false, false, true, true,
				true, false };
	}

	public static void main(String[] args) {
		List<String[]> trainDatas = DataConverter.simpleConvertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_train.txt"));

		List<String[]> testDatas = DataConverter.simpleConvertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_test.txt"));

		NB nb = new NB();
		nb.train(trainDatas);
		nb.test(testDatas);
	}

}
