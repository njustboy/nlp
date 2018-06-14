package com.ppp.dataminer.nlp.docclassify.tfidf;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ppp.dataminer.nlp.doc2vec.util.Word2Vec;


public class TFIDFUtil {
   /**
    * 表相加
    * 
    * @param map
    * @param addMap
    */
   public static void mapAdd(Map<String, Float> map, Map<String, Float> addMap) {
      for (String key : addMap.keySet()) {
         if (map.containsKey(key)) {
            map.put(key, map.get(key) + addMap.get(key));
         } else {
            map.put(key, addMap.get(key));
         }
      }
   }

   /**
    * 表相减
    * 
    * @param map
    * @param subMap
    */
   public static void mapSub(Map<String, Float> map, Map<String, Float> subMap) {
      for (String key : subMap.keySet()) {
         if (map.containsKey(key)) {
            float f = map.get(key) - subMap.get(key);
            if (f > 0) {
               map.put(key, f);
            } else {
               map.remove(key);
            }
         }
      }
   }

   /**
    * 计算分类离散系数
    * 
    * @param classifyMap
    * @return
    */
   public static float calcCV(Map<String, Float> classifyMap) {
      if (classifyMap.size() == 0) {
         return 0;
      }
      float average = 0f;
      float std = 0f;
      for (float value : classifyMap.values()) {
         average += value;
      }
      average /= classifyMap.size();

      for (float value : classifyMap.values()) {
         std += Math.pow(value - average, 2);
      }
      std = (float) Math.sqrt(std / classifyMap.size());

      return std / average;
   }

   /**
    * 通过计算词向量间的相似度计算词语的语义得分 使用的思想是：一个词和其他词越接近，则这个词的语义得分越高
    * 
    * @param list
    * @return
    */
   public static Map<String, Float> getSemanticScore(List<String> list) {
      Map<String, Float> map = new HashMap<String, Float>();
      for (String word : list) {
         float score = 0;
         for (String word1 : list) {
            if (!word.equals(word1)) {
               score += Word2Vec.getInstance().getDistance(word, word1);
            }
         }
         map.put(word, score);
      }
      return map;
   }
}
