package com.ppp.dataminer.nlp.topicmodel.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;

import com.ppp.dataminer.nlp.topicmodel.util.FileUtil;

/**
 * 文本集合
 * 
 * @author zhangwei
 *
 */
public class Documents {
	// 文本列表
	private List<Document> docs;
	// 词-->索引
	private Map<String, Integer> termToIndexMap;
	// 索引-->词
	private List<String> indexToTermMap;
	// 词-->个数
	private Map<String, Integer> termCountMap;
	// 将文本中的每一行作为一个doc处理
	private boolean dealLineAsFile = true;

	public Documents() {
		docs = new ArrayList<Document>();
		termToIndexMap = new HashMap<String, Integer>();
		indexToTermMap = new ArrayList<String>();
		termCountMap = new HashMap<String, Integer>();
		initStopWords();
	}

	public List<Document> getDocs() {
		return docs;
	}

	public Map<String, Integer> getTermToIndexMap() {
		return termToIndexMap;
	}

	public List<String> getIndexToTermMap() {
		return indexToTermMap;
	}

	public Map<String, Integer> getTermCountMap() {
		return termCountMap;
	}

	/**
	 * 读入文本，初始化文本--词矩阵
	 * 
	 * @param docsPath
	 */
	public void readDocs(String docsPath) {
		readFromFile(new File(docsPath));
	}

	private void readFromFile(File file) {
		if (file.isFile()) {
			if (dealLineAsFile) {
				// 文档中的每一行作为一个doc
				List<String> lines = new ArrayList<String>();
				FileUtil.readLines(file, lines);
				int maxCount = lines.size() > 1000 ? 1000 : lines.size();
				for (int i = 0; i < maxCount; i++) {
					Document doc = new Document(lines.get(i), termToIndexMap, indexToTermMap, termCountMap);
					docs.add(doc);
				}
			} else {
				Document doc = new Document(file, termToIndexMap, indexToTermMap, termCountMap);
				docs.add(doc);
			}
		} else if (file.isDirectory()) {
			File[] listFiles = file.listFiles();
			for (File listFile : listFiles) {
				readFromFile(listFile);
			}
		}
	}

	/**
	 * 文本数据类
	 * 
	 * @author zhangwei
	 *
	 */
	public static class Document {
		// wordindex(稀疏矩阵) 如5，6，7，5，3
		private int[] docWords;

		public int[] getDocWords() {
			return docWords;
		}

		/**
		 * 一个document为一个独立文本的情况
		 * 
		 * @param docName
		 * @param termToIndexMap
		 * @param indexToTermMap
		 * @param termCountMap
		 */
		public Document(File docFile, Map<String, Integer> termToIndexMap, List<String> indexToTermMap,
				Map<String, Integer> termCountMap) {
			// 词列表
			List<String> words = new ArrayList<String>();
			List<Term> parse = new ArrayList<Term>();
			try {
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(docFile), "utf8"));
				String line;
				int count = 0;
				while ((line = reader.readLine()) != null && count++ < 50) {
					parse = ToAnalysis.parse(line);
					// Remove stop words and noise words
					parse = FilterModifWord.modifResult(parse);
					for (Term term : parse) {
						if (term.getName().length() < 2) {
							continue;
						}
//						if (term.getName().contains("公司")||term.getName().contains("工作")||term.getName().contains("负责")) {
//							continue;
//						}
						words.add(term.getName());
					}
				}
				reader.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
			// 将word转为索引，这个索引是全局索引
			this.docWords = new int[words.size()];
			// 生成词索引
			for (int i = 0; i < words.size(); i++) {
				String word = words.get(i);
				if (!termToIndexMap.containsKey(word)) {
					int newIndex = termToIndexMap.size();
					termToIndexMap.put(word, newIndex);
					indexToTermMap.add(word);
					termCountMap.put(word, 1);
					docWords[i] = newIndex;
				} else {
					docWords[i] = termToIndexMap.get(word);
					termCountMap.put(word, termCountMap.get(word) + 1);
				}
			}
			words.clear();
		}

		/**
		 * 一个document对应一行文本的情况
		 * 
		 * @param content
		 * @param termToIndexMap
		 * @param indexToTermMap
		 * @param termCountMap
		 */
		public Document(String content, Map<String, Integer> termToIndexMap, List<String> indexToTermMap,
				Map<String, Integer> termCountMap) {
			List<Term> parse = ToAnalysis.parse(content);
			// Remove stop words and noise words
			parse = FilterModifWord.modifResult(parse);

			List<String> contentWords = new ArrayList<String>();
			for (int i = 0; i < parse.size(); i++) {
				String word = parse.get(i).getName();
				if (word.length() < 2) {
					continue;
				}
				contentWords.add(word);
			}

			// Transfer word to index
			this.docWords = new int[contentWords.size()];
			// 生成词索引
			for (int i = 0; i < contentWords.size(); i++) {
				String word = contentWords.get(i);
				if (!termToIndexMap.containsKey(word)) {
					int newIndex = termToIndexMap.size();
					termToIndexMap.put(word, newIndex);
					indexToTermMap.add(word);
					termCountMap.put(word, 1);
					docWords[i] = newIndex;
				} else {
					docWords[i] = termToIndexMap.get(word);
					termCountMap.put(word, termCountMap.get(word) + 1);
				}
			}
		}
	}

	private boolean initStopWords() {
		// init stop words
		File stopWordsFile = new File("library/stopword.dic");
		FileReader fr = null;
		BufferedReader br = null;
		String stopWord = null;
		try {
			fr = new FileReader(stopWordsFile);
			br = new BufferedReader(fr);
			stopWord = br.readLine();
			while (stopWord != null) {
				FilterModifWord.insertStopWord(stopWord);
				stopWord = br.readLine();
			}
			br.close();
			fr.close();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		} finally {
		}

		// init stop nature
		// File stopNaturesFile = new File("library/stopnature.dic");
		// String stopNature = null;
		// try {
		// fr = new FileReader(stopNaturesFile);
		// br = new BufferedReader(fr);
		// stopNature = br.readLine();
		// while (stopNature != null) {
		// FilterModifWord.insertStopNatures(stopNature);
		// stopNature = br.readLine();
		// }
		// br.close();
		// fr.close();
		// } catch (Exception e) {
		// e.printStackTrace();
		// return false;
		// } finally {
		// }
		return true;
	}

}
