package com.ppp.dataminer.nlp.doc2vec.train;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ppp.dataminer.nlp.doc2vec.data.Haffman;
import com.ppp.dataminer.nlp.doc2vec.data.HiddenNeuron;
import com.ppp.dataminer.nlp.doc2vec.data.Neuron;
import com.ppp.dataminer.nlp.doc2vec.data.WordNeuron;

/**
 * 文本向量和词向量训练器基类，实现haffman树的保存及读取操作，以支持模型的增量训练
 * 
 * 训练器使用层次softmax方法
 * 
 * @author zhangwei
 *
 */
public class TrainVec {
    // 词向量信息
    protected Map<String, Neuron> wordMap = new HashMap<>();

    // 词向量大小
    protected int layerSize = 100;

    // 上下文窗口大小
    protected int window = 5;

    protected double sample = 1e-3;
    // 迭代速率
    protected double alpha = 0.025;
    protected double startingAlpha = alpha;
    // sigmod函数静态表大小
    protected final int EXP_TABLE_SIZE = 1000;

    protected boolean isCbow = true;
    // sigmod函数静态表
    protected double[] expTable = new double[EXP_TABLE_SIZE];
    // 训练的总词数
    protected int trainWordsCount = 0;
    // 已训练的词汇数
    protected double wordCountActual = 0;
    // 使用sigmod函数时最大的自变量，当自变量超过此范围后sigmod函数值趋于0
    protected int MAX_EXP = 6;
    // 词频阈值
    protected int freqThresold = 5;
    // 训练迭代次数
    protected int iteratorNum = 20;

    protected Haffman haffman = null;

    /**
     * 从模型文件加载模型
     * 
     * @param modelFile
     */
    public TrainVec(File modelFile) {
        createExpTable();
        loadHaffmanModel(modelFile);
    }

    /**
     * 从内存加载模型
     * 
     * @param wordMap
     */
    public TrainVec(Map<String, Neuron> wordMap) {
        this.wordMap = wordMap;
        createExpTable();
    }

    public TrainVec() {
        createExpTable();
    }

    /**
     * 从模型文件加载haffman树
     * 
     * @param modelFile
     * @throws Exception
     */
    private void loadHaffmanModel(File modelFile) {
        Map<String, HiddenNeuron> hiddenNeurons = new HashMap<String, HiddenNeuron>();
        DataInputStream dis = null;
        try {
            dis = new DataInputStream(new BufferedInputStream(new FileInputStream(modelFile)));
            // 词向量数量
            dis.readInt();
            // 训练总词数
            trainWordsCount = dis.readInt();
            // 词向量长度
            layerSize = dis.readInt();

            // 当前层节点数
            int nodeSize = dis.readInt();
            // 循环读取每层节点数据
            while (nodeSize > 0) {
                for (int i = 0; i < nodeSize; i++) {
                    String nodeName = dis.readUTF();
                    // 自定义枝干节点以“node_”开头
                    if (nodeName.startsWith("node_")) {
                        // 隐层节点
                        HiddenNeuron hiddenNeuron = new HiddenNeuron();
                        hiddenNeuron.setName(nodeName);
                        hiddenNeuron.setParentName(dis.readUTF());
                        hiddenNeuron.setLeftChildName(dis.readUTF());
                        hiddenNeuron.setRightChildName(dis.readUTF());
                        float[] syn1 = new float[layerSize];
                        for (int j = 0; j < syn1.length; j++) {
                            syn1[j] = dis.readFloat();
                        }
                        hiddenNeuron.setSyn1(syn1);
                        // 读取分裂节点个数（此处始终为0）
                        dis.readInt();

                        hiddenNeurons.put(nodeName, hiddenNeuron);
                    } else {
                        // 词向量节点
                        WordNeuron wordNeuron = new WordNeuron();
                        wordNeuron.setName(nodeName);
                        wordNeuron.setParentName(dis.readUTF());
                        wordNeuron.setLeftChildName(dis.readUTF());
                        wordNeuron.setRightChildName(dis.readUTF());
                        float[] syn0 = new float[layerSize];
                        for (int j = 0; j < syn0.length; j++) {
                            syn0[j] = dis.readFloat();
                        }
                        wordNeuron.setSyn0(syn0);

                        int codeSize = dis.readInt();
                        int[] codeArr = new int[codeSize];
                        for (int j = 0; j < codeSize; j++) {
                            codeArr[j] = dis.readInt();
                        }
                        wordNeuron.setCodeArr(codeArr);

                        wordMap.put(nodeName, wordNeuron);
                    }
                }
                // 读取下一层节点个数
                nodeSize = dis.readInt();
            }
        } catch (Exception e) {
        } finally {
            if (dis != null) {
                try {
                    dis.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        // 构建词向量的路径链表
        for (Neuron wordNeuron : wordMap.values()) {
            List<Neuron> neurons = new ArrayList<Neuron>();
            String pName = wordNeuron.getParentName();
            while (!pName.equals("")) {
                Neuron pNode = hiddenNeurons.get(pName);
                neurons.add(pNode);
                pName = pNode.getParentName();
            }
            Collections.reverse(neurons);
            ((WordNeuron) wordNeuron).setNeurons(neurons);
        }

    }

    /**
     * 保存完整的haffman树，用于加载后二次训练
     * 
     * @param file
     */
    public void saveHoffman(File file) {
        List<Neuron> thisLevel = new ArrayList<Neuron>();
        List<Neuron> nextLevel = new ArrayList<Neuron>();
        Neuron root = haffman.getSet().first();
        thisLevel.add(root);
        DataOutputStream dataOutputStream = null;
        try {
            dataOutputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
            // 词向量数量
            dataOutputStream.writeInt(wordMap.size());
            // 训练总词数
            dataOutputStream.writeInt(trainWordsCount);
            // 词向量长度
            dataOutputStream.writeInt(layerSize);

            // 按层写入节点数据
            while (thisLevel.size() > 0) {
                // 写入当前层节点个数
                dataOutputStream.writeInt(thisLevel.size());
                for (Neuron neuron : thisLevel) {
                    if (neuron.getLeftChild() != null) {
                        nextLevel.add(neuron.getLeftChild());
                    }
                    if (neuron.getRightChild() != null) {
                        nextLevel.add(neuron.getRightChild());
                    }
                    // 当前节点名字
                    dataOutputStream.writeUTF(neuron.getName());
                    // 父节点名字
                    dataOutputStream.writeUTF(neuron.getParentName());
                    // 左子节点名字
                    dataOutputStream.writeUTF(neuron.getLeftChildName());
                    // 右子节点名字
                    dataOutputStream.writeUTF(neuron.getRightChildName());
                    // 节点向量
                    if (neuron instanceof HiddenNeuron) {
                        float[] syn1 = ((HiddenNeuron) neuron).getSyn1();
                        for (float d : syn1) {
                            dataOutputStream.writeFloat(d);
                        }
                    } else if (neuron instanceof WordNeuron) {
                        float[] syn0 = ((WordNeuron) neuron).getSyn0();
                        for (float d : syn0) {
                            dataOutputStream.writeFloat(d);
                        }
                    }
                    // 如果是词向量，还需记录分裂过程
                    if (neuron instanceof HiddenNeuron) {
                        dataOutputStream.writeInt(0);
                    } else if (neuron instanceof WordNeuron) {
                        int[] codeArr = ((WordNeuron) neuron).getCodeArr();
                        dataOutputStream.writeInt(codeArr.length);
                        for (int code : codeArr) {
                            dataOutputStream.writeInt(code);
                        }
                    }
                }

                thisLevel.clear();
                thisLevel.addAll(nextLevel);
                nextLevel.clear();
            }
            // 后边不再有节点
            dataOutputStream.writeInt(0);
            dataOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (dataOutputStream != null) {
                try {
                    dataOutputStream.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Precompute the exp() table f(x) = x / (x + 1)
     * 
     * y=(2i/1000-1)*6
     * 
     */
    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp((((double) 2 * i / EXP_TABLE_SIZE - 1) * MAX_EXP));
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public double getSample() {
        return sample;
    }

    public void setSample(double sample) {
        this.sample = sample;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public boolean getIsCbow() {
        return isCbow;
    }

    public void setIsCbow(boolean isCbow) {
        this.isCbow = isCbow;
    }

    public Map<String, Neuron> getWordMap() {
        return wordMap;
    }

    public void setWordMap(Map<String, Neuron> wordMap) {
        this.wordMap = wordMap;
    }

    public int getTrainWordsCount() {
        return trainWordsCount;
    }

    public void setTrainWordsCount(int trainWordsCount) {
        this.trainWordsCount = trainWordsCount;
    }

    public Haffman getHaffman() {
        return haffman;
    }

    public void setHaffman(Haffman haffman) {
        this.haffman = haffman;
    }

    public int getIteratorNum() {
        return iteratorNum;
    }

    public void setIteratorNum(int iteratorNum) {
        this.iteratorNum = iteratorNum;
    }
}
