package com.ppp.dataminer.nlp.doc2vec.data;

import java.util.Collection;
import java.util.TreeSet;

/**
 * haffman �� ������
 * 
 * @author zhangwei
 *
 */
public class Haffman {
    // ����������
    private int layerSize;
    // ���ڱ��haffman����֦�ɽڵ�
    private int count = 0;

    private TreeSet<Neuron> set = new TreeSet<>();

    public Haffman(int layerSize) {
        this.layerSize = layerSize;
    }

    /**
     * ������
     * @param neurons
     */
    public void make(Collection<Neuron> neurons) {
        set.addAll(neurons);
        while (set.size() > 1) {
            merger();
        }
    }

    /**
     * �ϲ�����������haffman�� �ϲ�ʱ��¼ÿ���ڵ�ĸ��ڵ㡢�����ӽڵ㣬�Լ���ؽڵ�����
     */
    private void merger() {
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        // ����ڵ�������node_��ͷ
        hn.setName("node_" + count);
        count++;
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        hn.setFreq(min1.getFreq() + min2.getFreq());
        min1.setParent(hn);
        min1.setParentName(hn.getName());
        min2.setParent(hn);
        min2.setParentName(hn.getName());
        min1.setCode(0);
        min2.setCode(1);
        hn.setLeftChild(min1);
        hn.setLeftChildName(min1.getName());
        hn.setRightChild(min2);
        hn.setRightChildName(min2.getName());
        set.add(hn);
    }

    public TreeSet<Neuron> getSet() {
        return set;
    }

    public void setSet(TreeSet<Neuron> set) {
        this.set = set;
    }
}
