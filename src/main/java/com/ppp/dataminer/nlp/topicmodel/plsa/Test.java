package com.ppp.dataminer.nlp.topicmodel.plsa;

import java.io.IOException;
import java.util.Arrays;

import com.ppp.dataminer.nlp.topicmodel.data.Documents;
import com.ppp.dataminer.nlp.topicmodel.data.ScoreComparable;

public class Test {
	public static void main(String[] args) throws IOException {
		 doTrain();
		 doInference("店员/营业员#&#&#在面料市场做营业员，每天像顾客推荐布料、款式，让他们能够购买商品。");
	}

	public static void doTrain() throws IOException {
		Documents docSet = new Documents();
		System.out.println("开始读取语料");
		docSet.readDocs("trainsource");
		System.out.println("文本数量: " + docSet.getDocs().size());
		System.out.println("词数量:" + docSet.getIndexToTermMap().size());
		PLSATraing model = new PLSATraing();
		System.out.println("初始化PLSA模型");
		model.initializeModel(docSet);
		long begin = System.currentTimeMillis();
		System.out.println("开始进行模型训练");
		model.learnModel(docSet);
		long end = System.currentTimeMillis();
		System.out.println("模型训练结束，共耗时：" + (end - begin) / 1000 + "秒");
		System.out.println("输出模型结果");
		model.saveIteratedModel(100, docSet);
		System.out.println("训练任务完成");
	}

	public static void doInference(String newDoc) {
		System.out.println("开始计算新文档主题分布");
		PLSAInference plsaInference = new PLSAInference();
		plsaInference.initializeModel("plsamodel/");
		long begin = System.currentTimeMillis();
		float[] topicPros = plsaInference.plsaInference(newDoc);
		// float[] topicPros = plsaInference.simplePlsaInference(newDoc);
		long end = System.currentTimeMillis();
		System.out.println("计算新文档主题分布共耗时：" + (end - begin) / 1000 + "秒");

		float sum = 0;
		for (float topicPro : topicPros) {
			sum += topicPro;
			System.out.print(topicPro + " ");
		}
		System.out.println();

		Integer[] index = new Integer[topicPros.length];
		for (int i = 0; i < index.length; i++) {
			index[i] = i;
		}
		Arrays.sort(index, new ScoreComparable(topicPros));

		System.out.println("主题总得分：" + sum);
		System.out.println("得分前3的主题索引及得分分别为：" + index[0] + ":" + topicPros[index[0]] + " " + index[1] + ":"
				+ topicPros[index[1]] + " " + index[2] + ":" + topicPros[index[2]]);
	}
}
