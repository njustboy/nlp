package nlp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.docclassify.vec.TFIDFVecClassifyApply;

public class SourceGenerater {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		generateWord2vecSource();
//		generateWords();
//		generateVocab();
		// System.out.println(System.currentTimeMillis());
		// Date data = new Date(1491926432000L);
		// System.out.println(data);

		// generateVecs();
//		generateFeatures();
	}

	private static void generateVecs() throws Exception {
		DecimalFormat df = new DecimalFormat("######0.00");
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		BufferedWriter bw = null;
		String line = null;
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("trainvecs/" + sourceFile.getName())));
			int count = 0;
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}
				double[] vecs = TFIDFVecClassifyApply.genTFIDFVec(segments[0], segments[1]);
				String writeLine = "";
				for (double d : vecs) {
					writeLine += df.format(d) + " ";
				}
				bw.write(writeLine);
				bw.newLine();
			}
			bw.flush();
			bw.close();
			br.close();
		}

		sourceDir = new File("trainsource");
		sourceFiles = sourceDir.listFiles();
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("testvecs/" + sourceFile.getName())));
			int count = 0;
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}
				double[] vecs = TFIDFVecClassifyApply.genTFIDFVec(segments[0], segments[1]);
				String writeLine = "";
				for (double d : vecs) {
					writeLine += df.format(d) + " ";
				}
				bw.write(writeLine);
				bw.newLine();
			}
			bw.flush();
			bw.close();
			br.close();
		}
	}

	private static void generateFeatures() throws Exception {
		Set<String> featureWords = new HashSet<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("model/txtFeatureDic")), "utf8"));;
		String line = null;
		while((line=br.readLine())!=null){
			featureWords.add(line.replaceAll(" ", "##"));
		}
		br.close();
		
		BufferedWriter bw = null;
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("trainfeatures/" + sourceFile.getName())));
			int count = 0;
			while ((line = br.readLine()) != null && count++ < 3000) {
				// tmpCount++;
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}
				
				segments[1] = segments[1].replaceAll(" ", "##");
				String writeLine = "";
				for(String featureWord:featureWords){
					while(segments[1].contains(featureWord)){
						writeLine += featureWord+" ";
						int beginIndex = segments[1].indexOf(featureWord);
						segments[1] = segments[1].substring(0, beginIndex)+segments[1].substring(beginIndex+featureWord.length());
					}
				}	

				if (writeLine.trim().length() > 0) {
					bw.write(writeLine);
					bw.newLine();
				}
			}
			bw.flush();
			bw.close();
			br.close();
		}

		sourceDir = new File("testsource");
		sourceFiles = sourceDir.listFiles();
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("testfeatures/" + sourceFile.getName())));
			int count = 0;
			while ((line = br.readLine()) != null && count++ < 1000) {
				// tmpCount++;
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}
				
				segments[1] = segments[1].replaceAll(" ", "##");
				String writeLine = "";
				for(String featureWord:featureWords){
					while(segments[1].contains(featureWord)){
						writeLine += featureWord+" ";
						int beginIndex = segments[1].indexOf(featureWord);
						segments[1] = segments[1].substring(0, beginIndex)+segments[1].substring(beginIndex+featureWord.length());
					}
				}	

				if (writeLine.trim().length() > 0) {
					bw.write(writeLine);
					bw.newLine();
				}
			}
			bw.flush();
			bw.close();
			br.close();
		}
	}
	
	private static void generateWords() throws Exception {
		initStopWords();
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		BufferedWriter bw = null;
		String line = null;
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("trainwords/" + sourceFile.getName())));
			int count = 0;
			// int tmpCount = 0;
			List<Term> tmpList = new ArrayList<Term>();
			while ((line = br.readLine()) != null && count++ < 3000) {
				// tmpCount++;
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}

				// 1、全量词汇
				List<Term> words = ToAnalysis.parse(segments[1]);
				words = FilterModifWord.modifResult(words);
				String writeLine = "";

				for (Term word : words) {
					if (word.getName().length() < 2) {
						continue;
					}
					writeLine += word.getName() + " ";
				}

				// 2、关键词
//				List<WordPair> parseKeywords = KeywordParserUtil.parseKeywords(segments[1], true);
//				for (int i = 0; i < parseKeywords.size() - 1; i++) {
//					for (int j = i + 1; j < parseKeywords.size(); j += 4) {
//						int endIndex = j + 3 > parseKeywords.size()-1 ? parseKeywords.size()-1 : j + 3;
//						writeLine += parseKeywords.get(i).getWord() + " ";
//						for (int k = j; k <= endIndex; k++) {
//							writeLine += parseKeywords.get(k).getWord() + " ";
//						}
//					}
//				}

				if (writeLine.trim().length() > 0) {
					bw.write(writeLine);
					bw.newLine();
				}
			}
			bw.flush();
			bw.close();
			br.close();
		}

		sourceDir = new File("testsource");
		sourceFiles = sourceDir.listFiles();
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			bw = new BufferedWriter(new FileWriter(new File("testwords/" + sourceFile.getName())));
			int count = 0;
			List<Term> tmpList = new ArrayList<Term>();
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("#&#&#");
				if (segments.length != 2) {
					continue;
				}

				// 1、全量词汇
				List<Term> words = ToAnalysis.parse(segments[1]);
				words = FilterModifWord.modifResult(words);
				String writeLine = "";

				for (Term word : words) {
					if (word.getName().length() < 2) {
						continue;
					}
					writeLine += word.getName() + " ";
				}

				// 2、关键词
//				List<WordPair> parseKeywords = KeywordParserUtil.parseKeywords(segments[1], true);
//				for (int i = 0; i < parseKeywords.size() - 1; i++) {
//					for (int j = i + 1; j < parseKeywords.size(); j += 4) {
//						int endIndex = j + 3 > parseKeywords.size()-1 ? parseKeywords.size()-1 : j + 3;
//						writeLine += parseKeywords.get(i).getWord() + " ";
//						for (int k = j; k <= endIndex; k++) {
//							writeLine += parseKeywords.get(k).getWord() + " ";
//						}
//					}
//				}
				
				if (writeLine.trim().length() > 0) {
					bw.write(writeLine);
					bw.newLine();
				}
				
				count ++;
				if(count>=1000){
					break;
				}
			}
			bw.flush();
			bw.close();
			br.close();
		}
	}

	private static boolean initStopWords() {
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
//		File stopNaturesFile = new File("library/stopnature.dic");
//		String stopNature = null;
//		try {
//			fr = new FileReader(stopNaturesFile);
//			br = new BufferedReader(fr);
//			stopNature = br.readLine();
//			while (stopNature != null) {
//				FilterModifWord.insertStopNatures(stopNature);
//				stopNature = br.readLine();
//			}
//			br.close();
//			fr.close();
//		} catch (Exception e) {
//			e.printStackTrace();
//			return false;
//		} finally {
//		}
		return true;
	}
	
	private static void generateVocab() throws Exception{
		Map<String, Map<String, Float>> tfidfMap = new HashMap<String, Map<String, Float>>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("model/tmptfidf")), "UTF-8"));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] segments = line.split("[\\{\\}]");
			if (segments.length == 2) {
				Map<String, Float> map = new HashMap<String, Float>();
				String[] wordPairs = segments[1].split(",");
				for (String wordPair : wordPairs) {
					String[] segs = wordPair.split("=");
					if (segs.length == 2) {
//						map.put(segs[0].trim(), (float)Math.log(Float.valueOf(segs[1].trim())));
						map.put(segs[0].trim(), Float.valueOf(segs[1].trim()));
					}
				}
				tfidfMap.put(segments[0].trim(), map);
			}
		}
		br.close();
		
		List<WordPair> list = new ArrayList<WordPair>();
		for(Entry<String,Map<String,Float>> entry:tfidfMap.entrySet()){
			List<Float> scoreList = new ArrayList<Float>(entry.getValue().values());
			while(scoreList.size()<60){
				scoreList.add(0f);
			}
			Float avg = 0f;
			for(Float score:scoreList){
				avg += score;
			}
			avg /= 60;
			Float f = 0f;
			for(float score:scoreList){
				f += Math.abs(score-avg);
			}
			list.add(new WordPair(entry.getKey(),f));
		}
		Collections.sort(list);
		
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("/Users/zhimatech/workspace/cnn-text-classification/model/vocab.txt"))));
		for(int i=0;i<5000;i++){
			bw.write(list.get(i).getWord());
			bw.newLine();
		}
		bw.close();
	}
	
	public static void generateWord2vecSource() throws Exception{
		initStopWords();
		File file = new File("/Users/zhimatech/Downloads/news_sohusite_xml.dat");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file),"gbk"));
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("corpus/word2vec/sougou_news.txt"))));
		String line = null;
		int count = 0;
		while((line=br.readLine())!=null){
			if(++count%10000==0){
				System.out.println("已处理"+count+"行");
			}
			// 只取正文
			if(!line.startsWith("<content>")){
				continue;
			}
			line = line.replaceAll("<content>", "").replaceAll("</content>", "");
			List<Term> list = ToAnalysis.parse(line);
			list = FilterModifWord.modifResult(list);
			for(Term term:list){
				bw.write(term.getName()+" ");
			}
		}
		br.close();
		bw.close();
	}

}
