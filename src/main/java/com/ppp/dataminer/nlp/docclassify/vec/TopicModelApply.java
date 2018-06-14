package com.ppp.dataminer.nlp.docclassify.vec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

/**
 * 主题模型应用
 * 
 * @author zhangwei
 *
 */
public class TopicModelApply {
	private static final int TOPIC_COUNT = 300;
	// word -->(topic,score)
	private static Map<String, Map<Integer, Double>> wordtopic = new HashMap<>();

	private static Map<String, Double> wordPercent = new HashMap<String, Double>();

	static {
		BufferedReader br = null;
		try {
			String line = null;
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("model/wordpercent")), "UTF-8"));
			while ((line = br.readLine()) != null) {
				String[] segments = line.split(":");
				if (segments.length == 2) {
					wordPercent.put(segments[0], Double.parseDouble(segments[1]));
				}
			}
			br.close();

			File file = new File("model/wordTopicDic");
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			while ((line=br.readLine()) != null) {
				line = line.replace("}", "");
				String[] s = line.split("\\{");
				if (s.length == 2) {
					Map<Integer, Double> map = new HashMap<Integer, Double>();
					String[] datas = s[1].split(",");
					for (String data : datas) {
						String[] keyvalue = data.split(":");
						if (keyvalue.length == 2) {
							map.put(Integer.parseInt(keyvalue[0].trim()),
									Double.parseDouble(keyvalue[1].trim())  / wordPercent.get(s[0]));
						}
					}
					if (map.size() > 0) {
						wordtopic.put(s[0].toLowerCase(), map);
					}
				}
				line = br.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (Exception ex) {

			}
		}
	}

	/**
	 * 将文本转化为主题分布向量
	 * 
	 * @param content
	 * @return
	 */
	public static double[] convertContent2vec(String content) {
		double[] vec = new double[TOPIC_COUNT];
		List<Term> parse = ToAnalysis.parse(content);
		// index-->score
		Map<Integer, Double> scoreMap = new HashMap<Integer, Double>();
		for (Term term : parse) {
			if (wordtopic.containsKey(term.getName())) {
				Map<Integer, Double> tmpMap = wordtopic.get(term.getName());
				for (Integer index : tmpMap.keySet()) {
					if (scoreMap.containsKey(index)) {
						scoreMap.put(index, scoreMap.get(index) + tmpMap.get(index));
					} else {
						scoreMap.put(index, tmpMap.get(index));
					}
				}
			}
		}
		double maxScore = 0;
		for (double score : scoreMap.values()) {
			if (score > maxScore) {
				maxScore = score;
			}
		}

		if (maxScore > 0) {
			for (int index : scoreMap.keySet()) {
				vec[index] = scoreMap.get(index) / maxScore;
				if (vec[index] < 0.0001) {
					vec[index] = 0;
				}
			}
		}

		return vec;
	}

	private static void analysisWordPercent() throws Exception {
		Set<String> words = wordtopic.keySet();
		Map<String, Double> map = new HashMap<String, Double>();
		File sourceFile = new File("trainwords/trainwords");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] segments = line.split("\\s+");
			for (String segmnet : segments) {
				if (words.contains(segmnet)) {
					if (map.containsKey(segmnet)) {
						map.put(segmnet, map.get(segmnet) + 1);
					} else {
						map.put(segmnet, 1.0);
					}
				}
			}
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("model/wordpercent")));
		for (Entry<String, Double> entry : map.entrySet()) {
			String outLine = entry.getKey() + ":" + entry.getValue();
			bw.write(outLine);
			bw.newLine();
		}
		bw.flush();
		bw.close();
	}

	public static void main(String[] args) throws Exception {
//		analysisWordPercent();
	}

}
