package com.ppp.dataminer.nlp.doc2vec.test;

import java.util.List;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.ppp.dataminer.nlp.doc2vec.util.Doc2Vec;

public class Doc2VecTest {

    public static void main(String[] args) {
        Doc2Vec doc2Vec = Doc2Vec.getInstance();
        List<Term> parse1 = ToAnalysis.parse("Ҫ�˽�������Ե����̣�������ԵĻ���֪ʶ����һЩ���õĲ��Եķ�����Ҫ����Ʊ�д������������ʹ�ù��ܻ������ܲ��Թ��ߣ��߼��Ժ󻹿��Ա�д�ű�����");
        String[] array1 = new String[parse1.size()];
        for(int i=0;i<parse1.size();i++){
            array1[i] = parse1.get(i).getName();
        }
        
        List<Term> parse2 = ToAnalysis.parse("1.���������������ƶ����Լƻ�����Ʋ������ݺͲ���������2.��Ч��ִ�в����������ύ���Ա��档3.׼ȷ�ض�λ���������⣬�ƶ����⼰ʱ����ؽ����4.��ɶԲ�Ʒ�ļ��ɲ�����ϵͳ���ԣ��Բ�Ʒ��������ܡ����ܼ���������Ĳ��ԡ�");
        String[] array2 = new String[parse2.size()];
        for(int i=0;i<parse2.size();i++){
            array2[i] = parse2.get(i).getName();
        }
        
        List<Term> parse3 = ToAnalysis.parse("������Ŀǰ̨���ܵĿ����Լ���Ԫ����,���μӲ������ݿ�ͺ�̨���ܵı�д");
        String[] array3 = new String[parse3.size()];
        for(int i=0;i<parse3.size();i++){
            array3[i] = parse3.get(i).getName();
        }
        
        System.out.println(doc2Vec.getDistance(array1, array2));
        System.out.println(doc2Vec.getDistance(array1, array3));
    }

}
