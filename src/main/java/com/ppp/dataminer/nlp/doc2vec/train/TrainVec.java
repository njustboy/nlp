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
 * �ı������ʹ�����ѵ�������࣬ʵ��haffman���ı��漰��ȡ��������֧��ģ�͵�����ѵ��
 * 
 * ѵ����ʹ�ò��softmax����
 * 
 * @author zhangwei
 *
 */
public class TrainVec {
    // ��������Ϣ
    protected Map<String, Neuron> wordMap = new HashMap<>();

    // ��������С
    protected int layerSize = 100;

    // �����Ĵ��ڴ�С
    protected int window = 5;

    protected double sample = 1e-3;
    // ��������
    protected double alpha = 0.025;
    protected double startingAlpha = alpha;
    // sigmod������̬���С
    protected final int EXP_TABLE_SIZE = 1000;

    protected boolean isCbow = true;
    // sigmod������̬��
    protected double[] expTable = new double[EXP_TABLE_SIZE];
    // ѵ�����ܴ���
    protected int trainWordsCount = 0;
    // ��ѵ���Ĵʻ���
    protected double wordCountActual = 0;
    // ʹ��sigmod����ʱ�����Ա��������Ա��������˷�Χ��sigmod����ֵ����0
    protected int MAX_EXP = 6;
    // ��Ƶ��ֵ
    protected int freqThresold = 5;
    // ѵ����������
    protected int iteratorNum = 20;

    protected Haffman haffman = null;

    /**
     * ��ģ���ļ�����ģ��
     * 
     * @param modelFile
     */
    public TrainVec(File modelFile) {
        createExpTable();
        loadHaffmanModel(modelFile);
    }

    /**
     * ���ڴ����ģ��
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
     * ��ģ���ļ�����haffman��
     * 
     * @param modelFile
     * @throws Exception
     */
    private void loadHaffmanModel(File modelFile) {
        Map<String, HiddenNeuron> hiddenNeurons = new HashMap<String, HiddenNeuron>();
        DataInputStream dis = null;
        try {
            dis = new DataInputStream(new BufferedInputStream(new FileInputStream(modelFile)));
            // ����������
            dis.readInt();
            // ѵ���ܴ���
            trainWordsCount = dis.readInt();
            // ����������
            layerSize = dis.readInt();

            // ��ǰ��ڵ���
            int nodeSize = dis.readInt();
            // ѭ����ȡÿ��ڵ�����
            while (nodeSize > 0) {
                for (int i = 0; i < nodeSize; i++) {
                    String nodeName = dis.readUTF();
                    // �Զ���֦�ɽڵ��ԡ�node_����ͷ
                    if (nodeName.startsWith("node_")) {
                        // ����ڵ�
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
                        // ��ȡ���ѽڵ�������˴�ʼ��Ϊ0��
                        dis.readInt();

                        hiddenNeurons.put(nodeName, hiddenNeuron);
                    } else {
                        // �������ڵ�
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
                // ��ȡ��һ��ڵ����
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

        // ������������·������
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
     * ����������haffman�������ڼ��غ����ѵ��
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
            // ����������
            dataOutputStream.writeInt(wordMap.size());
            // ѵ���ܴ���
            dataOutputStream.writeInt(trainWordsCount);
            // ����������
            dataOutputStream.writeInt(layerSize);

            // ����д��ڵ�����
            while (thisLevel.size() > 0) {
                // д�뵱ǰ��ڵ����
                dataOutputStream.writeInt(thisLevel.size());
                for (Neuron neuron : thisLevel) {
                    if (neuron.getLeftChild() != null) {
                        nextLevel.add(neuron.getLeftChild());
                    }
                    if (neuron.getRightChild() != null) {
                        nextLevel.add(neuron.getRightChild());
                    }
                    // ��ǰ�ڵ�����
                    dataOutputStream.writeUTF(neuron.getName());
                    // ���ڵ�����
                    dataOutputStream.writeUTF(neuron.getParentName());
                    // ���ӽڵ�����
                    dataOutputStream.writeUTF(neuron.getLeftChildName());
                    // ���ӽڵ�����
                    dataOutputStream.writeUTF(neuron.getRightChildName());
                    // �ڵ�����
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
                    // ����Ǵ������������¼���ѹ���
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
            // ��߲����нڵ�
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
