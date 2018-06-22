package com.ppp.dataminer.nlp.keyword;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;

public class KeyWordParserTest {

	public static void main(String[] args) {
		// String content =
		// "程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。";
		// String content =
		// "促销员#&#&#?要负责销售产品，与顾客交流，沟通。通过这次兼职，让我学到很多，知道要如何销售产品，怎么有效与顾客交流、沟通。同时，知道要怎么去提高自己的能力，达到自己的要求。";
		// String content =
		// "采购员#&#&#1.根据国外客户要求，通过各种渠道寻找国内优质供货商，陪同国外客户考察工厂，并确定采购，拟订采购合同，安排货运，收汇结汇等事宜。
		// 2.根据不同项目寻找合适的货代公司,降低公司物流成本. 3.KPI统计 4.制作进出口单据,联系货代确保货物准时出运";
//		String content = "本文首先简要回顾知识图谱的历史，探讨知识图谱研究的意义。其次，介绍知识图谱构建的关键技术，包括实体关系识别技术、知识融合技术、实体链接技术和知识推理技术等。然后，给出现有开放的知识图谱数据集的介绍。最后，给出知识图谱在情报分析中的应用案例。";
//		List<WordPair> keywords = KeywordParserUtil4Resume.parseKeywords(content, false);
//		System.out.println(keywords);
//		List<WordPair> keywords1 = KeywordParserUtil.parseKeywords(content, 10);
//		System.out.println(keywords1);
//		List<String> keywords2 = KeywordParserUtil.simpleParseKeywords(content, 10);
//		System.out.println(keywords2);
		
		test();

	}

	public static void test() {
		File dir = new File("testsource");
		File[] listFiles = dir.listFiles();
		BufferedReader br = null;
		for (File listFile : listFiles) {
			try {
				br = new BufferedReader(new InputStreamReader(new FileInputStream(listFile)));
				String line = null;
				int count = 0;
				while ((line = br.readLine()) != null && count++ < 50) {
					System.out.println(line);
					List<WordPair> keywords1 = KeywordParserUtil.parseKeywords(line, 10);
					System.out.println(keywords1);
					List<String> keywords2 = KeywordParserUtil.simpleParseKeywords(line, 10);
					System.out.println(keywords2);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

}
