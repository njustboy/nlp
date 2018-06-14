package com.ppp.dataminer.nlp.docclassify.tfidf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
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
import org.nlpcn.commons.lang.util.IOUtil;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;


/**
 * TFIDF杩唬璁粌
 * 
 * 鍙湪鏈湴绂荤嚎璁粌
 * 
 * @author zhangwei
 *
 */
public class TFIDFClassifyTrain {
   // 瀛︿範閫熺巼
   private static float alpha = 0.001f;
   // 鍒濆閫熺巼
   private static float startAlpha = 0.001f;
   // 杩唬娆℃暟
   private static int iterator = 5;
   // word-->(category,score)
   private static Map<String, Map<String, Float>> tfidfMap = new HashMap<String, Map<String, Float>>();

   static {
      BufferedReader br = null;
      String rootPath = "";

      try {
         br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(rootPath + "model/tmptfidf")), "UTF-8"));
         String line = null;
         while ((line = br.readLine()) != null) {
            String[] segments = line.split("[\\{\\}]");
            if (segments.length == 2) {
               Map<String, Float> map = new HashMap<String, Float>();
               String[] wordPairs = segments[1].split(",");
               for (String wordPair : wordPairs) {
                  String[] segs = wordPair.split("=");
                  if (segs.length == 2) {
                     map.put(segs[0].trim(), Float.valueOf(segs[1].trim()));
                  }
               }
               tfidfMap.put(segments[0].trim(), map);
            }
         }
      } catch (Exception e) {
      }
   }

   public static void main(String[] args) throws Exception {
      iteratorTrain(new File("/Users/zhimatech/workspace/cnn-text-classification/data/mrtrain"));
      IOUtil.writeMap(tfidfMap, "model/tfidf_new", "utf-8");
   }

   /**
    * 浣跨敤璇枡杩唬璁粌TFIDF鍊�
    * 
    * @param sourceDir
    * @throws Exception
    */
   public static void iteratorTrain(File sourceDir) throws Exception {
      BufferedReader br = null;
      String line = null;
      List<WordPair> classifyList = null;
      for (int i = 0; i < iterator; i++) {
         // 姣忔杩唬鍑忓皬瀛︿範姝ラ暱
         alpha = startAlpha * (iterator - i) / iterator;
         int totalCount = 0;
         int correctCount = 0;
         String analysisLine = "";
         for (File listFile : sourceDir.listFiles()) {
            // 鐪熷疄鍒嗙被鍗充负鏂囦欢鍚�
            String realClassify = listFile.getName();
            br = new BufferedReader(new FileReader(listFile));
            while ((line = br.readLine()) != null) {
               totalCount++;
               String[] segments = line.split("#&#&#");
               if (segments.length == 2) {
                  analysisLine = segments[0] + segments[0] + segments[1];
                  classifyList = docClassify(segments[0], segments[1]);
               } else {
                  analysisLine = line;
                  classifyList = docClassify("", line);
               }

               // 浠�涔堥兘娌″垎鍑烘潵锛岃繖鏄粈涔堟牱鐨勬枃鏈晩銆傘��
               if (classifyList.size() == 0) {
                  continue;
               }

               // 鍒嗙被瀹屽叏姝ｇ‘
               if (classifyList.get(0).getWord().equals(realClassify)) {
                  correctCount++;
                  // 澧炲己璇嶆眹瀵规纭垎绫荤殑寰楀垎
                  increase(analysisLine, realClassify);
                  continue;
               }

               // 鎵惧嚭姝ｇ‘鍒嗙被鍦ㄨ繑鍥炲垪琛ㄤ腑鐨勪綅缃�
               int index = 0;
               for (int j = 0; j < classifyList.size(); j++) {
                  if (classifyList.get(j).getWord().equals(realClassify)) {
                     index = j;
                     break;
                  }
               }
               // 姝ｇ‘鍒嗙被涓嶅湪杩斿洖鍒嗙被鍒楄〃涓紝琛ㄧず閿欑殑绂昏氨浜嗭紝鍙兘鏄牱鏈棶棰橈紝鐩存帴蹇界暐
               if (index == 0) {
                  continue;
               }

               // word-->score
               Map<String, Float> deltMap = null;
               for (int j = index - 1; j >= 0; j--) {
                  // 閿欒鍒嗙被
                  String classify = classifyList.get(j).getWord();
                  // 姣忎釜璇嶉渶瑕佷慨鏀圭殑寰楀垎
                  deltMap = calcDeltMap(analysisLine, classify, realClassify);

                  // 灏嗚宸洿鏂拌嚦ifidf琛ㄤ腑
                  for (String word : deltMap.keySet()) {
                     // 鏌愪釜璇嶅涓嶅悓鍒嗙被鐨勬敮鎸佸害
                     Map<String, Float> scoreMap = tfidfMap.get(word);
                     if (scoreMap.containsKey(classify)) {
                        // 鍑忓皬璇嶅閿欒鍒嗙被鐨勬敮鎸佸害
                        float score = scoreMap.get(classify) - deltMap.get(word);
                        if (score > 0) {
                           scoreMap.put(classify, score);
                        } else {
                           scoreMap.put(classify, scoreMap.get(classify) / 2);
                        }
                     }
                     // 鍔犲己璇嶅姝ｇ‘鍒嗙被鐨勬敮鎸佸害
                     float realScore = deltMap.get(word);
                     if (scoreMap.containsKey(realClassify)) {
                        realScore += scoreMap.get(realClassify);
                        scoreMap.put(realClassify, realScore);
                     }
                  }
               }
            }
         }
         System.out.println("绗�" + (i + 1) + "娆¤凯浠ｇ粨鏉燂紝鍏辨湁鏍锋湰" + totalCount + "鏉★紝鍒嗙被姝ｇ‘" + correctCount + "鏉★紝鍒嗙被姝ｇ‘鐜�"
               + correctCount / (double) totalCount);
      }
   }

   /**
    * 
    * @param line
    *           鍒嗙被鏂囨湰
    * @param classify
    *           閿欒鍒嗙被
    * @param realClassify
    *           姝ｇ‘鍒嗙被
    * @return
    */
   private static Map<String, Float> calcDeltMap(String line, String classify, String realClassify) {
      Map<String, Float> deltMap = new HashMap<String, Float>();
      List<Term> parse = ToAnalysis.parse(line);
      Set<String> wordSet = new HashSet<String>();
      for (Term term : parse) {
         wordSet.add(term.getName());
      }
      for (String word : wordSet) {
         if (!tfidfMap.containsKey(word)) {
            continue;
         }
         // 鏌愪釜璇嶅鍚勫垎绫绘敮鎸佸害
         Map<String, Float> map = tfidfMap.get(word);
         float f1 = 0f;
         float f2 = 0f;
         if (map.containsKey(classify)) {
            f1 = map.get(classify);
         }
         if (map.containsKey(realClassify)) {
            f2 = map.get(realClassify);
         }
         // 濡傛灉鏌愪釜璇嶅閿欒鍒嗙被鐨勬敮鎸佸害澶т簬姝ｇ‘鍒嗙被鐨勬敮鎸佸害锛屽垯闇�瑕佽繘琛屼慨姝�
         // 鍦ㄨ繖閲岃涓哄垵濮嬬殑缁熻缁撴灉鏄瘮杈冮潬璋辩殑锛屽鏋渇1杩滃ぇ浜巉2锛屽垯涓嶅鐞�
         if (f1 > f2 && f1 < 2 * f2) {
            deltMap.put(word, (f1 - f2) * alpha);
         } else if (f2 > f1) {
            // 澧炲己姝ｇ‘寰楀垎
            map.put(realClassify, map.get(realClassify) * (1 + alpha * 0.01f));
         }
      }
      return deltMap;
   }

   /**
    * 澧炲姞鏂囨湰涓瘝姹囧鏌愪釜鍒嗙被鐨勬敮鎸佽
    * 
    * @param line
    * @param realClassify
    */
   private static void increase(String line, String realClassify) {
      List<Term> parse = ToAnalysis.parse(line);
      Set<String> wordSet = new HashSet<String>();
      for (Term term : parse) {
         wordSet.add(term.getName());
      }
      for (String word : wordSet) {
         if (!tfidfMap.containsKey(word)) {
            continue;
         }
         Map<String, Float> scoreMap = tfidfMap.get(word);
         if (scoreMap.containsKey(realClassify)) {
            scoreMap.put(realClassify, scoreMap.get(realClassify) * (1 + alpha * 0.01f));
         }
      }
   }

   /**
    * 浣跨敤鏂囨湰涓瘝姹囩殑TFIDF鍊艰繘琛屾枃鏈垎绫�
    * 
    * @param position
    * @param content
    * @return
    */
   private static List<WordPair> docClassify(String position, String content) {
      List<WordPair> classifyList = new ArrayList<WordPair>();

      Map<String, Float> scoreMap = new HashMap<String, Float>();
      List<Term> parse = ToAnalysis.parse(position + position + content);
      for (Term term : parse) {
         if (tfidfMap.containsKey(term.getName())) {
            TFIDFUtil.mapAdd(scoreMap, tfidfMap.get(term.getName()));
         }
      }

      for (String name : scoreMap.keySet()) {
         classifyList.add(new WordPair(name, scoreMap.get(name)));
      }
      Collections.sort(classifyList);

      return classifyList;
   }
}
