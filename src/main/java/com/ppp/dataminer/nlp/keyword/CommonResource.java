package com.ppp.dataminer.nlp.keyword;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 提供一些公共资源的类 IDF=log(n/N),n表示包含关键词的文档数，N表示总文档数 Word2VEC,用于查找相似词、相关词
 * 
 * @author zhangwei
 *
 */
public class CommonResource {

   private static CommonResource commonResource = null;

   public static CommonResource getInstance() {
      if (null == commonResource) {
         synchronized (CommonResource.class) {
            if (null == commonResource) {
               commonResource = new CommonResource();
            }
         }
      }

      return commonResource;
   }

   // 简历描述的关键词的IDF信息
   private volatile Map<String, Double> workexpIDF = null;

   // 职位描述的关键词的IDF信息
   private volatile Map<String, Double> jobexpIDF = null;
   // 职位名称关键字列表
   private List<String> jobtitleWords = null;

   // 停用词性列表
   private List<String> stopNature = null;

   // notkeyword
   private List<String> notKeyword = null;

   private CommonResource() {

   }

   /**
    * 返回简历工作描述关键词的IDF值
    * 
    * @return
    */
   public Map<String, Double> getWorkexpIDF() {
      if (null == workexpIDF) {
         synchronized (CommonResource.class) {
            if (null == workexpIDF) {
               initWorkexpIDF();
            }
         }
      }
      return workexpIDF;
   }

   /**
    * 返回职位要求关键词的IDF值
    * 
    * @return
    */
   public Map<String, Double> getjobexpIDF() {
      if (null == jobexpIDF) {
         synchronized (CommonResource.class) {
            if (null == jobexpIDF) {
               initJobexpIDF();
            }
         }
      }
      return jobexpIDF;
   }

   /**
    * 返回职位名关键词列表
    * 
    * @return
    */
   public List<String> getJobtitleWords() {
      if (jobtitleWords == null) {
         initJobtitleWords();
      }
      return jobtitleWords;
   }

   /**
    * 返回停用词性列表
    * 
    * @return
    */
   public List<String> getStopNature() {
      if (null == stopNature) {
         initStopNature();
      }
      return stopNature;

   }

   /**
    * 返回非关键字列表
    * 
    * @return
    */
   public List<String> getNotKeyword() {
      if (notKeyword == null) {
         initNotKeyword();
      }
      return notKeyword;
   }

   /**
    * 初始啥非关键字列表
    */
   private void initNotKeyword() {
      notKeyword = new ArrayList<String>();
      String rootPath = "";
      File file = new File("library/notkeyword.dic");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(file);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         String line = br.readLine();
         while (line != null) {

            notKeyword.add(line);
            line = br.readLine();
         }
      } catch (Exception e) {
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }

   }

   /**
    * 初始化简历工作描述关键词的IDF值
    */
   private void initWorkexpIDF() {
      workexpIDF = new HashMap<String, Double>();
      File file = new File("dic/workexpIDF");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(file);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         String line = br.readLine();
         while (line != null) {
            String[] strs = line.split(":");
            workexpIDF.put(strs[0], Double.parseDouble(strs[1]));
            line = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }
   }

   /**
    * 初始化职位要求关键词的IDF值
    */
   private void initJobexpIDF() {
      jobexpIDF = new HashMap<String, Double>();
      File file = new File("dic/jobexpIDF");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(file);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         String line = br.readLine();
         while (line != null) {
            String[] strs = line.split(":");
            jobexpIDF.put(strs[0], Double.parseDouble(strs[1]));
            line = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }
   }

   /**
    * 初始化职位名关键词列表
    */
   private void initJobtitleWords() {
      jobtitleWords = new ArrayList<String>();
      File file = new File("dic/jobtitleWords");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(file);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);

         String line = br.readLine();
         while (line != null) {
            jobtitleWords.add(line);
            line = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }
   }

   /**
    * 初始化停用词性列表
    */
   private void initStopNature() {
      stopNature = new ArrayList<String>();
      String line = null;
      File file = new File("library/stopnature.dic");
      FileInputStream fis = null;
      InputStreamReader isr = null;
      BufferedReader br = null;
      try {
         fis = new FileInputStream(file);
         isr = new InputStreamReader(fis, "utf-8");
         br = new BufferedReader(isr);
         line = br.readLine();
         while (line != null) {
            stopNature.add(line);
            line = br.readLine();
         }
      } catch (Exception e) {
         e.printStackTrace();
      } finally {
         try{
        	 br.close();
         }catch(Exception ex){
        	 ex.printStackTrace();
         }
      }

   }

}
