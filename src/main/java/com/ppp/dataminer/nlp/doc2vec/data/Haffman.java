package com.ppp.dataminer.nlp.doc2vec.data;

import java.util.Collection;
import java.util.TreeSet;

/**
 * haffman 树 数据类
 * 
 * @author zhangwei
 *
 */
public class Haffman {
    // 词向量长度
    private int layerSize;
    // 用于标记haffman树中枝干节点
    private int count = 0;

    private TreeSet<Neuron> set = new TreeSet<>();

    public Haffman(int layerSize) {
        this.layerSize = layerSize;
    }

    /**
     * 构造树
     * @param neurons
     */
    public void make(Collection<Neuron> neurons) {
        set.addAll(neurons);
        while (set.size() > 1) {
            merger();
        }
    }

    /**
     * 合并词向量生成haffman树 合并时记录每个节点的父节点、左右子节点，以及相关节点名称
     */
    private void merger() {
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        // 隐层节点名称用node_开头
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
