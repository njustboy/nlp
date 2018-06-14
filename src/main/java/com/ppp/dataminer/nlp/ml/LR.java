package com.ppp.dataminer.nlp.ml;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * 逻辑回归简单实现
 * 
 * @author zhimatech
 *
 */
public class LR {
	private int iterator = 100;

	private double theater = 1;

	private double w0 = 0;

	private double[] w;

	private Map<Integer, Double> sigmodMap = null;

	/**
	 * 训练过程
	 * 
	 * @param trainDatas
	 * @param trainLabels
	 */
	public void train(double[][] trainDatas, int[] trainLabels) {
		if (trainDatas.length == 0) {
			System.out.println("there is no train data");
			return;
		}

		initW(trainDatas[0].length);

		double flag = 0;

		double thisTheater = theater / trainDatas.length;

		// 迭代训练
		for (int i = 0; i < iterator; i++) {
			if (i % 10 == 0) {
				System.out.println("迭代训练" + i + "次");
				test(trainDatas, trainLabels);
			}
			double deltaW0 = 0;
			double[] deltaW = new double[w.length];

			// 缓存每条数据的分类结果
			Map<Integer, Double> tmpMap = new HashMap<Integer, Double>();
			for (int k = 0; k < trainDatas.length; k++) {
				tmpMap.put(k, sigmod(trainDatas[k]) - trainLabels[k]);
			}

			// 计算w0更新项
			double sum = 0;
			for (int k = 0; k < trainDatas.length; k++) {
				sum += tmpMap.get(k);
			}
			deltaW0 = thisTheater * sum;

			// 计算每个参数的更新项
			for (int j = 0; j < w.length; j++) {
				sum = 0;

				for (int k = 0; k < trainDatas.length; k++) {
					sum += tmpMap.get(k) * trainDatas[k][j];
				}

				deltaW[j] = thisTheater * sum;
			}

			System.out.print(deltaW0 + " ");
			for (double delta : deltaW) {
				System.out.print(delta + " ");
			}
			System.out.println();

			// 更新参数
			w0 -= deltaW0;
			for (int j = 0; j < w.length; j++) {
				w[j] -= deltaW[j];
			}

			flag += Math.abs(deltaW0);
			for (int j = 0; j < w.length; j++) {
				flag += Math.abs(deltaW[j]);
			}
			// 如果参数更新趋于平稳，则结束迭代
			if (flag < 0.01) {
				// break;
			}

			tmpMap.clear();
		}

	}

	/**
	 * 测试过程
	 * 
	 * @param testDatas
	 * @param testLabels
	 */
	public void test(double[][] testDatas, int[] testLabels) {
		int correctNum = 0;

		for (int i = 0; i < testDatas.length; i++) {
			double score = sigmod(testDatas[i]);

			if (score >= 0.5 && testLabels[i] == 1) {
				correctNum++;
			} else if (score < 0.5 && testLabels[i] == 0) {
				correctNum++;
			}
		}

		System.out.println("total test num:" + testDatas.length);
		System.out.println("total correct num:" + correctNum);
		System.out.println("correct rate :" + 1.0 * correctNum / testDatas.length);
	}

	private void initW(int length) {
		w = new double[length];
	}

	private double sigmod(double[] data) {
		if (sigmodMap == null) {
			initSigmodMap();
		}
		
		double score = -w0;

		for (int i = 0; i < data.length; i++) {
			score -= data[i] * w[i];
		}

		int id = (int) (100 * score);
		if (id > 600) {
			score = sigmodMap.get(600);
		} else if (id < -600) {
			score = sigmodMap.get(-600);
		} else {
			score = sigmodMap.get(id);
		}

		return score;
	}

	private void initSigmodMap() {
		sigmodMap = new HashMap<Integer, Double>();
		for (int i = -600; i <= 600; i++) {
			double d = i / 100.0;
			sigmodMap.put(i, 1 / (1 + Math.pow(Math.E, d)));
		}
	}

	public static void main(String[] args) {
		Map<Integer, Object> trainMap = DataConverter.convertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_train.txt"), null,true);
		System.out.println("训练数据构造完毕");
		Map<Integer, Object> testMap = DataConverter.convertAdultData(
				new File("/Users/zhimatech/Documents/机器学习_NB_LR算法实现+实验报告/程序+数据样本+测试结果/adult_test.txt"), null,false);
		System.out.println("测试数据构造完毕");
		LR lr = new LR();
		long begin = System.currentTimeMillis();
		lr.train((double[][]) trainMap.get(1), (int[]) trainMap.get(2));
		long end = System.currentTimeMillis();
		System.out.println("训练用时：" + (end - begin) / 1000 + "秒");
		lr.test((double[][]) testMap.get(1), (int[]) testMap.get(2));
	}

}
