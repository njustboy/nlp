package com.ppp.dataminer.nlp.doc2vec.test;

import java.util.List;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.doc2vec.util.Doc2Vec;

public class Doc2VecTest {

    public static void main(String[] args) {
        Doc2Vec doc2Vec = Doc2Vec.getInstance();
        List<Term> parse1 = ToAnalysis.parse("要了解软件测试的流程，软件测试的基础知识。会一些常用的测试的方法，要会设计编写测试用例，会使用功能或者性能测试工具，高级以后还可以编写脚本语言");
        String[] array1 = new String[parse1.size()];
        for(int i=0;i<parse1.size();i++){
            array1[i] = parse1.get(i).getName();
        }
        
        List<Term> parse2 = ToAnalysis.parse("1.根据软件设计需求制定测试计划，设计测试数据和测试用例；2.有效地执行测试用例，提交测试报告。3.准确地定位并跟踪问题，推动问题及时合理地解决；4.完成对产品的集成测试与系统测试，对产品的软件功能、性能及其它方面的测试。");
        String[] array2 = new String[parse2.size()];
        for(int i=0;i<parse2.size();i++){
            array2[i] = parse2.get(i).getName();
        }
        
        List<Term> parse3 = ToAnalysis.parse("负责项目前台功能的开发以及单元测试,还参加部分数据库和后台功能的编写");
        String[] array3 = new String[parse3.size()];
        for(int i=0;i<parse3.size();i++){
            array3[i] = parse3.get(i).getName();
        }
        
        System.out.println(doc2Vec.getDistance(array1, array2));
        System.out.println(doc2Vec.getDistance(array1, array3));
    }

}
