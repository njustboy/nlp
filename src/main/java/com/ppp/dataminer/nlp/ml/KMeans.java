package com.ppp.dataminer.nlp.ml;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * k-means简单实现
 * 
 * @author zhimatech
 *
 */
public class KMeans {
	private int kernelCount = 50;

	private double[][] centers;

	private int iterator = 200;

	private void train(double[][] data) {
		// 初始化中心点
		initCenter(data);
		// 记录每一条数据所属的类别
		int[] clusterIdArray = new int[data.length];

		for (int i = 0; i < iterator; i++) {
			Map<Integer, Map<Integer, Double>> tmpMap = new HashMap<Integer, Map<Integer, Double>>();
			for (int l = 0; l < centers.length; l++) {
				Map<Integer, Double> subMap = new HashMap<Integer, Double>();
				for (int m = 0; m < centers.length; m++) {
					subMap.put(m, distance(centers[l], centers[m]));
				}
				tmpMap.put(l, subMap);
			}

			for (int j = 0; j < data.length; j++) {
				 double minDistance = Double.MAX_VALUE;
				 int clusterId = 0;
				 for (int k = 0; k < centers.length; k++) {
				 double tmpDistance = distance(data[j], centers[k]);
				 if (tmpDistance < minDistance) {
				 minDistance = tmpDistance;
				 clusterId = k;
				 }
				 }

				// 距离上一个聚类中心点的距离
//				int clusterId = clusterIdArray[j];
//				double minDistance = distance(data[j], centers[clusterId]);
//				for (int k = 0; k < centers.length; k++) {
//					if (k == clusterId) {
//						continue;
//					}
//					if (tmpMap.get(k).get(clusterId) > 2 * minDistance) {
//						continue;
//					}
//					
//					double tmpDistance = distance(data[j],centers[k]);
//					if(tmpDistance<minDistance){
//						minDistance = tmpDistance;
//						clusterId = k;
//					}
//				}

				clusterIdArray[j] = clusterId;
			}

			// 重新计算中心点
			reCaleCenter(data, clusterIdArray);
			// 计算所有数据到各自中心点的平均距离
//			double avgDistance = calcAvgDistance(data, clusterIdArray, centers);

//			System.out.println("第" + (i + 1) + "次迭代，平均距离为：" + avgDistance);
		}
		
		double avgDistance = calcAvgDistance(data, clusterIdArray, centers);
		System.out.println("平均距离为："+avgDistance);

	}

	private double calcAvgDistance(double[][] data, int[] clusterIdArray, double[][] centers) {
		double totalDistance = 0;

		for (int i = 0; i < data.length; i++) {
			totalDistance += distance(data[i], centers[clusterIdArray[i]]);
		}

		double avgDistance = totalDistance / data.length;

		return avgDistance;
	}

	private int[] reCaleCenter(double[][] data, int[] clusterIdArray) {
		int[] counts = new int[kernelCount];
		double[][] tmpCenters = new double[kernelCount][centers[0].length];

		for (int i = 0; i < data.length; i++) {
			double[] thisData = data[i];
			counts[clusterIdArray[i]]++;
			for (int j = 0; j < thisData.length; j++) {
				tmpCenters[clusterIdArray[i]][j] += thisData[j];
			}
		}

		for (int i = 0; i < centers.length; i++) {
			for (int j = 0; j < centers[0].length; j++) {
				centers[i][j] = tmpCenters[i][j] / counts[i];
			}
		}

		return counts;
	}

	private void initCenter(double[][] data) {
		centers = new double[kernelCount][];
		// 取第一条数据作为第一个中心点
		centers[0] = data[0];
		Set<Integer> idSet = new HashSet<Integer>();
		idSet.add(0);
		for (int i = 1; i < kernelCount; i++) {
			double maxDistance = 0;
			int thisId = 0;
			for (int j = 1; j < data.length; j++) {
				if (idSet.contains(j)) {
					continue;
				}
				// 找出和已有中心点的最近距离
				double minDistance = Double.MAX_VALUE;
				for (Integer id : idSet) {
					double thisDistance = distance(data[j], data[id]);
					if (thisDistance < minDistance) {
						minDistance = thisDistance;
					}
				}

				// 找出和所有中心点距离最远的点
				if (minDistance > maxDistance) {
					maxDistance = minDistance;
					thisId = j;
				}
			}

			idSet.add(thisId);
			centers[i] = data[thisId];
		}
	}

	private double distance(double[] var1, double[] var2) {
		double d = 0;
		for (int i = 0; i < var1.length; i++) {
			d += Math.pow(var1[i] - var2[i], 2);
		}
		return Math.pow(d, 0.5);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Map<Integer, Object> trainMap = DataConverter.convertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_train.txt"), null, true);

		KMeans km = new KMeans();
		long begin = System.currentTimeMillis();
		km.train((double[][]) trainMap.get(1));
		long end = System.currentTimeMillis();
		System.out.println("训练耗时：" + (end - begin) / 1000 + "秒");
	}

}
