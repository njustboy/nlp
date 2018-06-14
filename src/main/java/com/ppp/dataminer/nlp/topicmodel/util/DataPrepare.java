package com.ppp.dataminer.nlp.topicmodel.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.List;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;

public class DataPrepare {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		initStopWords();
		prepareData();
	}

	private static void prepareData() throws Exception {
		File sourceDir = new File("trainsource");
		File[] sourceFiles = sourceDir.listFiles();
		BufferedReader br = null;
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("trainwords/trainwords")));
		for (File sourceFile : sourceFiles) {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(sourceFile), "utf8"));
			String line = null;
			while ((line = br.readLine()) != null) {
				List<Term> parse = ToAnalysis.parse(line);
				// Remove stop words and noise words
				parse = FilterModifWord.modifResult(parse);
				String outLine = "";
				for (Term term : parse) {
					outLine += term.getName() + " ";
				}
				bw.write(outLine);
				bw.newLine();
			}
			br.close();
		}
		bw.flush();
		bw.close();
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
