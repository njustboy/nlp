package com.ppp.dataminer.nlp.docclassify.tfidf;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

public class Test {

   public static void main(String[] args) throws Exception{
      List<Term> parse = ToAnalysis.parse("熟悉APP、为对方提供ASO咨询冲榜、刷榜等");
      System.out.println(parse);
      
   }

}
