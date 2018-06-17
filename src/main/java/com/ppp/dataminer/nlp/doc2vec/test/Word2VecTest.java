package com.ppp.dataminer.nlp.doc2vec.test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;
import com.ppp.dataminer.nlp.doc2vec.util.Word2Vec;

public class Word2VecTest {
    public static void main(String[] args) throws Exception{
//        // 迭代训练次数
//        int iteratorNum = 15;
//        // 词向量长度
//        int vecLen = 200;
//        try{
//           iteratorNum = Integer.parseInt(args[0]);
//           vecLen = Integer.parseInt(args[1]);
//        }catch(Exception e){
//           
//        }
//        TrainWordVec twv = new TrainWordVec();
//        twv.setIteratorNum(iteratorNum);
//        twv.setLayerSize(vecLen);
//        long begin = System.currentTimeMillis();
//        twv.learnFile(new File("corpus/word2vec"));
//        twv.saveHoffman(new File("model/haffman_"+iteratorNum+"_"+vecLen+".mod"));
//        twv.saveWordVecs(new File("model/wordvec_"+iteratorNum+"_"+vecLen+".bin"));
//        long end = System.currentTimeMillis();
//        System.out.println("训练模型共耗时："+(end-begin)/1000+"秒");
        
        Word2Vec w2v = Word2Vec.getInstance();
        
        Set<String> apmacs = w2v.getWords();
        for(String apmac:apmacs){
        	Set<WordPair> simWords = w2v.distance(apmac);
        	String line = apmac+":";
        	for(WordPair wp:simWords){
        		line += wp.getWord()+"_"+wp.getWeight()+",";
        	}
        	System.out.println(line.substring(0, line.length()-1));
        }
       
//        System.out.println(w2v.distance("java"));
//        System.out.println(w2v.distance("php"));
//        System.out.println(w2v.distance("房地产"));
//        System.out.println(w2v.distance("销售"));
//        System.out.println(w2v.distance("咖啡"));
//        System.out.println(w2v.distance("机械"));
//        System.out.println(w2v.distance("公务员"));
//        System.out.println(w2v.distance("化学"));
//        System.out.println(w2v.distance("导游"));
//        System.out.println(w2v.distance("化妆"));
//        System.out.println(w2v.distance("仓库"));
//        System.out.println(w2v.distance("薪资"));
//        System.out.println(w2v.distance("宝马"));
//  
//        System.out.println(w2v.analogy("java", "ssh", "php"));
     }
    
    private static boolean isSame(String str1,String str2){
    	boolean isSame = false;
    	str1 = str1.replaceAll("^\\u4e00-\\u9fa5", "");
    	str2 = str2.replaceAll("^\\u4e00-\\u9fa5", "");
    	List<Term> list1 = ToAnalysis.parse(str1);
    	List<Term> list2 = ToAnalysis.parse(str2);
    	for(Term term1:list1){
    		for(Term term2:list2){
    			if(term1.getName().equals(term2.getName())){
    				isSame = true;
    				break;
    			}
    		}
    	}
    	return isSame;
    }
}
