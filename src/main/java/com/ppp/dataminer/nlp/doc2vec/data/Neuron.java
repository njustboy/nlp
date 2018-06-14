package com.ppp.dataminer.nlp.doc2vec.data;

public class Neuron implements Comparable<Neuron> {
    private String name = "";
    // ��Ԫ����Ƶ�ʣ�����ָ�ʳ���Ƶ��
    private int freq;
    // ���ڵ�
    private Neuron parent;
    // ���ڵ�����
    private String parentName = "";
    // ��ڵ�
    private Neuron leftChild;
    // �ҽڵ�
    private Neuron rightChild;
    // ��ڵ�����
    private String leftChildName = "";
    // �ҽڵ�����
    private String rightChildName = "";
    // ��ʾ���ҽڵ�
    private int code;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getFreq() {
        return freq;
    }

    public void setFreq(int freq) {
        this.freq = freq;
    }

    public Neuron getParent() {
        return parent;
    }

    public void setParent(Neuron parent) {
        this.parent = parent;
    }

    public String getParentName() {
        return parentName;
    }

    public void setParentName(String parentName) {
        this.parentName = parentName;
    }

    public Neuron getLeftChild() {
        return leftChild;
    }

    public void setLeftChild(Neuron leftChild) {
        this.leftChild = leftChild;
    }

    public Neuron getRightChild() {
        return rightChild;
    }

    public void setRightChild(Neuron rightChild) {
        this.rightChild = rightChild;
    }

    public String getLeftChildName() {
        return leftChildName;
    }

    public void setLeftChildName(String leftChildName) {
        this.leftChildName = leftChildName;
    }

    public String getRightChildName() {
        return rightChildName;
    }

    public void setRightChildName(String rightChildName) {
        this.rightChildName = rightChildName;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    
    public int compareTo(Neuron o) {
        // TODO Auto-generated method stub
        if (this.freq > o.freq) {
            return 1;
        } else {
            return -1;
        }
    }
}
