package com.ppp.dataminer.nlp.keyword;

import java.util.Set;

import com.ppp.dataminer.nlp.doc2vec.util.Word2Vec;
public class SimilarWordUtil {
   /**
    *  分析一个词对于文本主题到得分  
    * @param word
    * @param set
    * @return
    */
   public static double getSemanticScore(String word,Set<String> set){
      double score = 0;
      for(String simWord:set){
         score += Word2Vec.getInstance().getSimility(word,simWord);
      }
      return score;
   }
}
