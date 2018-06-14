package com.ppp.dataminer.nlp.docclassify.tfidf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.nlpcn.commons.lang.util.IOUtil;

/**
 * 基础TF-IDF表生成工具
 * 
 * @author zhangwei
 *
 */
public class ITIDFGenerater {
   private static DecimalFormat df = new DecimalFormat("0.000");

   public static void main(String[] args) throws Exception {
	   generateTFIDF("trainwords","model/tmptfidf");
   }

   /**
    * 通过给定样本统计词汇TFIDF表
    * 语料需要先进行分词，不同分类到语料放在不同文件中
    * @param sourcePath
    *           预料路径
    * @param modelPath
    *           输出文件路径
    * @throws Exception
    */
   public static void generateTFIDF(String sourcePath, String modelPath) throws Exception {
      // word-->(category,score)
      Map<String, Map<String, String>> tfidfMap = new HashMap<String, Map<String, String>>();
      // word-->(category,count)
      Map<String, Map<String, Integer>> wordCountMap = new HashMap<String, Map<String, Integer>>();
      // category-->wordcount
      Map<String, Double> categoryWordCount = new HashMap<String, Double>();

      Map<String,Integer> categoryCount = new HashMap<String,Integer>();
      
      File sourceDir = new File(sourcePath);

      for (File listFile : sourceDir.listFiles()) {
         // 文件名即为分类名
         String fileName = listFile.getName();
         BufferedReader br = new BufferedReader(new FileReader(listFile));
         String line = null;
         int count = 0;
         while ((line = br.readLine()) != null) {
        	 count++;
//        	line = cleanString(line);
            String[] words = line.split("\\s+");
            List<String> wordsList = new ArrayList<String>();
            for(String word: words){
            	if(word.length()>1){
            		wordsList.add(word);
            	}
            }
            wordsList.toArray(words);
            if (categoryWordCount.containsKey(fileName)) {
               categoryWordCount.put(fileName, categoryWordCount.get(fileName) + words.length);
            } else {
               categoryWordCount.put(fileName, (double) words.length);
            }

            for (String word : words) {
               Map<String, Integer> map = null;
               if (wordCountMap.containsKey(word)) {
                  map = wordCountMap.get(word);
               } else {
                  map = new HashMap<String, Integer>();
               }

               if (map.containsKey(fileName)) {
                  map.put(fileName, map.get(fileName) + 1);
               } else {
                  map.put(fileName, 1);
               }

               wordCountMap.put(word, map);
            }
         }
         categoryCount.put(fileName, count);
         br.close();
      }

      // 词--分类--数量
      for (Entry<String, Map<String, Integer>> entry : wordCountMap.entrySet()) {
         String word = entry.getKey();
         Map<String, Integer> map = entry.getValue();

         int totalCount = 0;
         for (Integer ii : map.values()) {
            totalCount += ii;
         }
         // 过滤词频太小的词
         if (totalCount < 10) {
            continue;
         }

         double dd = 0;
         for (String category : map.keySet()) {
            dd += map.get(category) / categoryWordCount.get(category);
         }
         dd = 60 / dd;

         Map<String, String> IDFMap = new HashMap<String, String>();
         for (String category : map.keySet()) {
            IDFMap.put(category, df.format(map.get(category) / categoryWordCount.get(category)*100000));
//            IDFMap.put(category, df.format(map.get(category) / categoryWordCount.get(category)*dd));
         }
         tfidfMap.put(word, IDFMap);
      }
      tfidfMap.remove(null);
      IOUtil.writeMap(tfidfMap, modelPath, "utf-8");
      IOUtil.writeMap(categoryCount, "model/categoryCount.txt", "utf-8");
      
   }
   
   private static String cleanString(String str){
	   str = str.replaceAll("[^A-Za-z0-9(),!?\'\\`]", " ");
	   str = str.replaceAll("\\'s", " \\'s");
	   str = str.replaceAll("\\'ve", " \\'ve");
	   str = str.replaceAll("n\\'t", " n\\'t");
	   str = str.replaceAll("\\'re", " \\'re");
	   str = str.replaceAll("\\'d", " \\'d");
	   str = str.replaceAll("\\'ll", " \\'ll");
	   str = str.replaceAll(",", " , ");
	   str = str.replaceAll("!", " ! ");
	   str = str.replaceAll("\\(", " \\( ");
	   str = str.replaceAll("\\)", " \\) ");
	   str = str.replaceAll("\\?", " \\? ");
	   str = str.replaceAll("\\s{2,}", " ");
	   return str;
   }

}
