package com.ppp.dataminer.nlp.doc2vec.data;

import java.util.Random;

/**
 * haffman树枝干节点
 * 
 * @author zhangwei
 *
 */
public class HiddenNeuron extends Neuron {
    // 神经元向量标示，对应haffman树枝干节点向量
    private float[] syn1;
    // 从根节点到此节点的二叉分裂标志
    private int[] codeArr = null;

    public float[] getSyn1() {
        return syn1;
    }

    public void setSyn1(float[] syn1) {
        this.syn1 = syn1;
    }

    public int[] getCodeArr() {
        return codeArr;
    }

    public void setCodeArr(int[] codeArr) {
        this.codeArr = codeArr;
    }

    public HiddenNeuron(int layerSize) {
        if (layerSize > 0) {
            syn1 = new float[layerSize];
            Random random = new Random();
            for (int i = 0; i < syn1.length; i++) {
                syn1[i] = (float) (random.nextFloat() - 0.5) / layerSize;
            }
        }
    }

    public HiddenNeuron() {

    }
}
