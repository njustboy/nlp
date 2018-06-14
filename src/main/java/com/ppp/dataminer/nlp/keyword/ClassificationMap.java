package com.ppp.dataminer.nlp.keyword;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * 职能分类映射关系表
 * 
 * @author zhangwei
 *
 */
public class ClassificationMap {
   // 二级类到一级类的映射
   private static Map<String, String> tow2oneMap = new HashMap<String, String>();
   // 三级类到一级类的映射
   private static Map<String, String> three2oneMap = new HashMap<String, String>();
   static {
      File two2oneFile = new File("dic/classificationMap");
      File three2oneFile = new File("dic/jobtitles");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(two2oneFile);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         String line = br.readLine();
         while (line != null) {
            String[] segments = line.split("=");
            if (segments.length != 2) {
               line = br.readLine();
               continue;
            }
            String two = segments[0].replaceAll("-", "\\/").trim();
            String one = segments[1].replaceAll("-", "\\/").trim().replaceFirst("\\/", "");
            tow2oneMap.put(two, one);
            line = br.readLine();
         }

         fis = new FileInputStream(three2oneFile);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         line = br.readLine();
         String one = "";
         while (line != null) {
            if (line.startsWith("title:")) {
               one = line.replace("title:", "");
               line = br.readLine();
               continue;
            }
            three2oneMap.put(line, one);
            line = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
      }finally{
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }
   }

   /**
    * 获取从二级到一级的名称
    * 
    * @param two
    * @return
    */
   public static String twoToOne(String two) {
      if (tow2oneMap.containsKey(two)) {
         return tow2oneMap.get(two);
      }
      return "";
   }

   /**
    * 获取从二级到一级的名称
    * 
    * @param two
    * @return
    */
   public static String threeToOne(String three) {
      if (three2oneMap.containsKey(three)) {
         return three2oneMap.get(three);
      }
      return "";
   }
}
