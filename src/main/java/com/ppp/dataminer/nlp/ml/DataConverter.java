package com.ppp.dataminer.nlp.ml;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 将原始数据集转化为算法能够使用的输入数据的工具类
 * 
 * @author zhimatech
 *
 */
public class DataConverter {
	// 非数值特征的范围列表
	private static Map<Integer, List<String>> notNumFeatures = new HashMap<Integer, List<String>>();
	// 数值特征的平均值
	private static Map<Integer, Double> numFeatures = new HashMap<Integer, Double>();

	/**
	 * 读取adult数据集
	 * 
	 * @param file
	 * @return
	 */
	public static Map<Integer, Object> convertAdultData(File file, File outFile, boolean recognizeFaultData) {
		Map<Integer, Object> adultData = new HashMap<Integer, Object>();
		List<String[]> dataList = new ArrayList<String[]>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
			String line = null;
			String[] segments = null;
			// 数据读取
			while ((line = br.readLine()) != null) {
				segments = line.split(",");
				dataList.add(segments);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// 标记某个特征是否为数值型
		boolean[] featureIsNum = new boolean[] { true, false, true, false, true, false, false, false, false, false,
				true, true, true, false };

		if (numFeatures.size() == 0) {
			// 统计各特征的取值范围
			for (int k = 0; k < dataList.size() - 1; k++) {
				String[] data = dataList.get(k);
				for (int i = 0; i < data.length - 1; i++) {
					if (featureIsNum[i]) {
						if (numFeatures.containsKey(i)) {
							numFeatures.put(i, numFeatures.get(i) + Double.parseDouble(data[i]));
						} else {
							numFeatures.put(i, Double.parseDouble(data[i]));
						}
					} else {
						if (notNumFeatures.containsKey(i)) {
							List<String> features = notNumFeatures.get(i);
							if (!features.contains(data[i])) {
								features.add(data[i]);
							}
						} else {
							List<String> features = new ArrayList<String>();
							features.add(data[i]);
							notNumFeatures.put(i, features);
						}
					}
				}
			}

			for (Integer id : notNumFeatures.keySet()) {
				Collections.sort(notNumFeatures.get(id));
				System.out.println(id + ":" + notNumFeatures.get(id));
			}
			for (Integer id : numFeatures.keySet()) {
				numFeatures.put(id, numFeatures.get(id) / dataList.size());
				System.out.println(id + ":" + numFeatures.get(id));
			}
		}

		// 特征数据
		double[][] datas = new double[dataList.size()][];
		// 标签
		int[] lables = new int[dataList.size()];

		// 构造的特征长度
		int featureLength = 0;
		featureLength += numFeatures.size();
		for (Integer id : notNumFeatures.keySet()) {
			featureLength += notNumFeatures.get(id).size();
		}

		for (int i = 0; i < dataList.size(); i++) {
			String[] data = dataList.get(i);
			double[] features = new double[featureLength];
			int index = 0;
			boolean isFault = false;
			for (int j = 0; j < data.length - 1; j++) {
				if (featureIsNum[j]) {
					try {
						features[index] = Double.parseDouble(data[j]) / numFeatures.get(j);

						if (features[index] > 5 || (features[index] < 0.2 && Double.parseDouble(data[j]) > 0)) {
							// 认为是异常值，需要去除
							// System.out.println("发现异常数据");
							isFault = true;
							break;
						}

					} catch (Exception e) {
						// 如果发生异常则用平均值代替
						features[index] = 1;
					}
				} else {
					int id = notNumFeatures.get(j).indexOf(data[j]);
					id = id < 0 ? 0 : id;
					features[index + id] = 1.0;
					index += notNumFeatures.get(j).size();
				}
			}
			if (isFault && recognizeFaultData) {
				// 如果第一条数据就是异常数据，这里会出现异常。。。
				datas[i] = datas[i - 1];
				lables[i] = lables[i - 1];
			} else {
				datas[i] = features;
				lables[i] = data[data.length - 1].trim().contains("<=50K") ? 0 : 1;
			}
		}

		adultData.put(1, datas);
		adultData.put(2, lables);

		System.out.println("特征数量：" + datas[0].length);

		// try {
		// BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
		// new FileOutputStream(outFile)));
		// for (int i = 0; i < datas.length; i++) {
		// String outLine = "";
		// for (double feature : datas[i]) {
		// outLine += feature + ",";
		// }
		// outLine += lables[i];
		// bw.write(outLine);
		// bw.newLine();
		// }
		// bw.close();
		//
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
		return adultData;
	}

	public static List<String[]> simpleConvertAdultData(File file) {
		List<String[]> dataList = new ArrayList<String[]>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
			String line = null;
			String[] segments = null;
			// 数据读取
			while ((line = br.readLine()) != null) {
				// if(line.contains("?")){
				// continue;
				// }
				segments = line.split(",");
				if (segments.length != 15) {
					// System.out.println(line);
					continue;
				}
				if (segments[segments.length - 1].contains("<=50K")) {
					segments[segments.length - 1] = "0";
				} else {
					segments[segments.length - 1] = "1";
				}
				dataList.add(segments);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return dataList;
	}

	public static void main(String[] args) {
		DataConverter.convertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_train.txt"),
				new File("/Users/zhimatech/workspace/testpy/train.csv"), true);
		DataConverter.convertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_test.txt"),
				new File("/Users/zhimatech/workspace/testpy/test.csv"), false);
	}
}
