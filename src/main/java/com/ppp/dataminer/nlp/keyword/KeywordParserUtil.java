package com.ppp.dataminer.nlp.keyword;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.docclassify.tfidf.TFIDFClassifyApply;

public class KeywordParserUtil {
   /**
    * 静态生成词性并赋予不同词性不同重要性
    */
   public static Map<String, Double> POS_SCORE = new HashMap<String, Double>();
   static {
      POS_SCORE.put("null", 0.0);
      POS_SCORE.put("w", 0.0);
      POS_SCORE.put("en", 5.0);
      POS_SCORE.put("num", 0.0);
      POS_SCORE.put("nr", 0.0);
      POS_SCORE.put("n", 2.5);
      POS_SCORE.put("nrf", 2.5);
      POS_SCORE.put("nw", 2.5);
      POS_SCORE.put("nt", 2.5);
      POS_SCORE.put("l", 1.0);
      POS_SCORE.put("a", 0.5);
      POS_SCORE.put("nz", 1.0);
      POS_SCORE.put("v", 1.0);
      POS_SCORE.put("vn", 1.5);
      // 自定义词性应该给与高权重
      POS_SCORE.put("userDefine", 5.0);
      initStopwords();
   }

   /**
    * 
    * @param term
    * @param length
    * @return 由词性生成关键词的词性权重
    */
   private static double getweightcixing(Term term, int length) {
      if (term.getName().trim().length() < 2) {
         return 0;
      }
      String pos = term.natrue().natureStr;
      Double posScore = POS_SCORE.get(pos);

      if (posScore == null) {
         posScore = 1.0;
      } else if (posScore == 0) {
         return 0;
      }
      return (length - term.getOffe()) * posScore / (double) length;
   }

   /**
    * 
    * @param tempMap
    * @param tempcixing
    * @return 将关键字的TFIDF与CIXING重要性组合成新的关键字权重
    */
   private static List<WordPair> update(Map<String, Double> tempMap, Map<String, Double> tempcixing) {
      List<WordPair> returnlist = new ArrayList<WordPair>();

      Set<String> keyset = tempMap.keySet();
      double truevalue = 0.0;
      for (String ky : keyset) {
         Double mapvalue1 = tempMap.get(ky);
         Double cixingvalue2 = tempcixing.get(ky);
         // 处理一方不包含对应关键字的情况，value值域为（0，1）之间，赋值应该在此区间

         if (cixingvalue2 == null || cixingvalue2.isNaN()) {
            cixingvalue2 = Config.UPDATE_WHEN_NULL;
         }
         truevalue = Config.TFIDF_PERCENT * mapvalue1 + Config.CIXING_PERCENT * cixingvalue2;

         WordPair wp = new WordPair();
         wp.setWeight(truevalue);
         wp.setWord(ky);
         returnlist.add(wp);
      }

      return returnlist;
   }
   /**
    * 
    * @param tempMap 词频得分表
    * @param tempcixing 词性得分表
    * @param tmpSemanticMap 语义得分表
    * @param tmpTopicMap 主题得分表
    * @return
    */
	private static List<WordPair> update(Map<String, Double> tempMap, Map<String, Double> tempcixing,Map<String,Double>tmpSemanticMap,Map<String,Double>tmpTopicMap) {
      List<WordPair> returnlist = new ArrayList<WordPair>();

      Set<String> keyset = tempMap.keySet();
      double truevalue = 0.0;
      for (String ky : keyset) {
         Double mapvalue = tempMap.get(ky);
         Double cixingvalue = tempcixing.get(ky);
         Double semanticValue = tmpSemanticMap.get(ky);
         Double topicValue = tmpTopicMap.get(ky);
         // 处理一方不包含对应关键字的情况，value值域为（0，1）之间，赋值应该在此区间
         if (cixingvalue == null || cixingvalue.isNaN()) {
            cixingvalue = Config.UPDATE_WHEN_NULL;
         }
         if(semanticValue==null||semanticValue.isNaN()){
            semanticValue = Config.UPDATE_WHEN_NULL;
         }
         if(topicValue==null||topicValue.isNaN()){
            topicValue = Config.UPDATE_WHEN_NULL;
         }
         
         truevalue = Config.TFIDF_PERCENT* mapvalue + Config.CIXING_PERCENT * cixingvalue+Config.SEMANTIC_PERCENT*semanticValue+Config.TOPIC_PERCENT*topicValue;

         WordPair wp = new WordPair();
         wp.setWeight(truevalue);
         wp.setWord(ky);
         returnlist.add(wp);
      }

      return returnlist;
   }

   /**
    * 
    * @param tempMap
    * @return 归一化value值，消除组间误差
    */
   private static Map<String, Double> normalized(Map<String, Double> tempMap) {
      Set<String> keyset = tempMap.keySet();
      double maxvalue = 0.0;
      double minvalue = 9999.0;
      for (String word : keyset) {

         maxvalue = tempMap.get(word) > maxvalue ? tempMap.get(word) : maxvalue;
         minvalue = tempMap.get(word) < minvalue ? tempMap.get(word) : minvalue;
      }
      double dvalue = maxvalue - minvalue;
      double truevalue = 0.0;
      if (dvalue > 0) {
         for (String wd : keyset) {
            truevalue = (tempMap.get(wd) - minvalue) / dvalue;
            tempMap.put(wd, truevalue);
         }
      } else {
         for (String wd : keyset) {
            tempMap.put(wd, 1.0);
         }
      }
      return tempMap;
   }

   /**
    * 
    * @return 初始化停用此表（包括停用词与停用词性）
    */
   private static boolean initStopwords() {
      File stopWordsFile = new File("library/stopword.dic");
      String stopWord = null;
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(stopWordsFile);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         stopWord = br.readLine();
         while (stopWord != null) {
            FilterModifWord.insertStopWord(stopWord);
            stopWord = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
         return false;
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }
      return true;
   }

   /**
    * 去停用词性
    * 
    * @param wordlist
    * @param stopNature
    * @return
    */
   private static List<Term> modifStopNature(List<Term> wordlist, List<String> stopNature) {
      List<Term> rewordlist = new ArrayList<Term>();
      for (Term term : wordlist) {
         if (stopNature.contains(term.getNatureStr())) {
            continue;
         }
         rewordlist.add(term);
      }
      return rewordlist;
   }

   /**
    * 去非关键字
    * 
    * @param wordlist
    * @param notKeyword
    * @return
    */
   private static List<Term> modifNotKeyword(List<Term> wordlist, List<String> notKeyword) {
      List<Term> rewordlist = new ArrayList<Term>();
      for (Term term : wordlist) {
         if (notKeyword.contains(term.getName())) {
            continue;
         }
         rewordlist.add(term);
      }
      return rewordlist;

   }

   /**
    * 
    * @param content
    * @param (boolean)isFromResume
    * @return 提供提取关键字及其权重接口，返回list，内部数据格式为WordPair
    */
   public static List<WordPair> parseKeywords(String content, boolean isFromResume) {
      List<String> categories = new ArrayList<String>();
      return parseKeywords(content, categories, isFromResume);
   }

   public static List<WordPair> parseKeywords(String content,String category, boolean isFromResume) {
      List<String> categories = new ArrayList<String>();
      categories.add(category);
      return parseKeywords(content, categories, isFromResume);
   }
   
   /**
    * 计算关键词权重时会考虑关键词对分类的支持度
    * 
    * @param content
    * @param (boolean)isFromResume
    * @return 提供提取关键字及其权重接口，返回list，内部数据格式为WordPair
    */
   public static List<WordPair> parseKeywords(String content, List<String> categories, boolean isFromResume) {

      double IDF_max = 10.0;
      // 读入IDF
      Map<String, Double> IDFMap = null;
      // 判断需要解析的文本来自简历还是来自职位，二者的IDF不同
      if (isFromResume) {
         IDFMap = CommonResource.getInstance().getWorkexpIDF();
         IDF_max = Config.WORKIDF_MAX;
      } else {
         IDFMap = CommonResource.getInstance().getjobexpIDF();
         IDF_max = Config.JOBIDF_MAX;
      }

      // 职位名关键词列表
      List<String> jobtitleWords = CommonResource.getInstance().getJobtitleWords();
      // 停用词性列表
      List<String> stopNature = CommonResource.getInstance().getStopNature();

      // 存储非关键字字典列表
      List<String> notKeyword = CommonResource.getInstance().getNotKeyword();

      // 存储返回所的关键字列表
      List<WordPair> rekeyword = new ArrayList<WordPair>();

      // 分词
      content = content.toLowerCase();
      List<Term> wordlist = ToAnalysis.parse(content);
      // 去停用词
      wordlist = FilterModifWord.modifResult(wordlist);
      // 去停用词性
//      wordlist = modifStopNature(wordlist, stopNature);
      // 去非关键字列表
//      wordlist = modifNotKeyword(wordlist, notKeyword);
      
      // 操作TFIDF主要用到的MAP
      Map<String, Double> tmpMap = new HashMap<String, Double>();
      // 存储词性关键字及权重的列表
      Map<String, Double> tempcixing = new HashMap<String, Double>();
      // 语义得分
      Map<String,Double> tmpSemanticMap = new HashMap<String,Double>();
      // 主题支持度得分
      Map<String,Double> tmpTopicMap = new HashMap<String,Double>();
      // 字符去重
      Set<String> wordSet = new HashSet<String>();
      // 关键词分类支持度分值表
      Map<String, Map<String, Float>> scoreMap = TFIDFClassifyApply.tfidfMap;
      for (Term term : wordlist) {
         // 关键词不能是单个的字
         if (term.getName() == null || term.getName().length() < 2) {
            continue;
         }
         // 记录出现的词
         wordSet.add(term.getName());
         // 计算关键字TF
         if (tmpMap.containsKey(term.getName())) {
            tmpMap.put(term.getName(), tmpMap.get(term.getName()) + 1);
         } else {
            tmpMap.put(term.getName(), 1.0);
         }

         // 计算词性关键字及权重
         double weightcixing = getweightcixing(term, content.length());
         if (weightcixing == 0) {
            continue;
         }
         // 更新词性得分
         if(tempcixing.containsKey(term.getName())){
            tempcixing.put(term.getName(), tempcixing.get(term.getName())+weightcixing);
         }else{
            tempcixing.put(term.getName(), weightcixing);
         }
      }
      
     // 计算主题支持度及分类支持度得分
      for(String word:wordSet){
         // 主题支持度得分
         double semanticScore = SimilarWordUtil.getSemanticScore(word, wordSet);
         tmpSemanticMap.put(word, semanticScore);
         
         // 分类支持度得分
         if(!scoreMap.containsKey(word)){
            // 未被统计的词给0分
            tmpTopicMap.put(word, 0.0);
         }else{
         // 词对于各个二级分类的支持度
            Map<String, Float> map = scoreMap.get(word);
            // 词对于各个一级分类的支持度
            Map<String, Double> oneMap = new HashMap<String, Double>();
            for (String key : map.keySet()) {
               // 获取一级职能
               String position = ClassificationMap.twoToOne(key.replaceAll("_", "/"));
               if (oneMap.containsKey(position)) {
                  oneMap.put(position, oneMap.get(position) + map.get(key));
               } else {
                  oneMap.put(position, (double)map.get(key));
               }
            }
            // 这个词对于当前分类的支持度
            double topicScore = 0.0;
            for (String category : categories) {
               if (oneMap.containsKey(category)) {
                  topicScore += oneMap.get(category);
               }
            }

            if (topicScore == 0) {
               // 这个词对分类没有支持的情况
               tmpTopicMap.put(word, 0.0);
            } else {
               // 这个词对所有分类的总体支持度
               double sumScore = 0.0;
               for (Double d : oneMap.values()) {
                  sumScore += d;
               }
               tmpTopicMap.put(word, topicScore / sumScore);
            }
         }
      }

      // TF结合IDF构成关键字的权重
      Set<String> keySet = tmpMap.keySet();
      for (String word : keySet) {
         // tmpMap里面存储的是关键字与其TF
         double weight = (IDF_max / 2) * tmpMap.get(word);
         if (IDFMap.containsKey(word)) {
            double idf = IDFMap.get(word);
            if (jobtitleWords.contains(word)) {
               idf = idf * Config.JOBTITLE_WEIGHT;
            }
            weight = Math.pow(idf, Config.IDF_WEIGHT) * tmpMap.get(word);
         } else if (tempcixing.containsKey(word)) {
            weight = Math.min(tempcixing.get(word), (IDF_max / 2) * tmpMap.get(word));
         }
         // 更新关键字的值更新:TF->TFIDF
         tmpMap.put(word, weight);
      }

      // 词性关键字与TFIDF关键字归一化后互相组合
      tmpMap = normalized(tmpMap);
      tempcixing = normalized(tempcixing);
      tmpSemanticMap = normalized(tmpSemanticMap);
      tmpTopicMap = normalized(tmpTopicMap);
      
      rekeyword = update(tmpMap, tempcixing,tmpSemanticMap,tmpTopicMap);
     
      // 排序，取前KEYWORD_MAXCOUNT个关键字，少于则返回本身
      Collections.sort(rekeyword);
//      print("词频+词性+分类支持度+主題支持度：",rekeyword);
      
      List<WordPair> re = new ArrayList<WordPair>();
      if (rekeyword.size() > Config.KEYWORD_MAXCOUNT) {
         for (int i = 0; i < Config.KEYWORD_MAXCOUNT; i++) {
            re.add(rekeyword.get(i));
         }
         return re;
      } else {
         return rekeyword;
      }
   }
   
   private static void print(String head,List<WordPair> list){
      int maxIndex = list.size()>8?8:list.size();
      System.out.print(head);
      for(int i=0;i<maxIndex;i++){
         System.out.print(list.get(i).getWord()+" ");
      }
      System.out.println();
   }
}
