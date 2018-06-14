package com.ppp.dataminer.nlp.doc2vec.data;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class WordNeuron extends Neuron {
    // 节点向量，即词向量
    private float[] syn0 = null;
    // 路径神经元
    private List<Neuron> neurons = null;
    // 从根节点到此节点的二叉分裂标志
    private int[] codeArr = null;

    public float[] getSyn0() {
        return syn0;
    }

    public void setSyn0(float[] syn0) {
        this.syn0 = syn0;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    public int[] getCodeArr() {
        return codeArr;
    }

    public void setCodeArr(int[] codeArr) {
        this.codeArr = codeArr;
    }

    /**
     * 生成从根节点到叶子的路径和分裂记录
     * @return
     */
    public List<Neuron> makeNeurons() {
        if (neurons != null) {
            return neurons;
        }
        Neuron neuron = this;
        neurons = new LinkedList<>();
        while ((neuron = neuron.getParent()) != null) {
            neurons.add(neuron);
        }
        // haffman树路径
        Collections.reverse(neurons);
        codeArr = new int[neurons.size()];
        // 分裂标记
        for (int i = 1; i < neurons.size(); i++) {
            codeArr[i - 1] = neurons.get(i).getCode();
        }
        codeArr[codeArr.length - 1] = this.getCode();

        return neurons;
    }

    public WordNeuron(String name, int freq, int layerSize) {
        setName(name);
        setFreq(freq);
        if (layerSize > 0) {
            this.syn0 = new float[layerSize];
            Random random = new Random();
            for (int i = 0; i < syn0.length; i++) {
                syn0[i] = (float) (random.nextFloat() - 0.5) / layerSize;
            }
        }
    }

    public WordNeuron() {

    }
}
