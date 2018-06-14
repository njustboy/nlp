package com.ppp.dataminer.nlp.doc2vec.train;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.ppp.dataminer.nlp.doc2vec.data.Haffman;
import com.ppp.dataminer.nlp.doc2vec.data.HiddenNeuron;
import com.ppp.dataminer.nlp.doc2vec.data.Neuron;
import com.ppp.dataminer.nlp.doc2vec.data.WordNeuron;

/**
 * ������ѵ������ ������Ҫ���÷ִʹ��߷ִʣ��ʻ���ÿո����
 * 
 * @author zhangwei
 *
 */
public class TrainWordVec extends TrainVec {
    /**
     * ��ģ���ļ�����ģ��
     * 
     * @param modelFile
     */
    public TrainWordVec(File modelFile) {
        super(modelFile);
    }

    /**
     * ���ڴ����ģ��
     * 
     * @param wordMap
     */
    public TrainWordVec(Map<String, Neuron> wordMap) {
        super(wordMap);
    }

    public TrainWordVec() {
        super();
    }

    /**
     * �����ļ�ѧϰ
     * 
     * @param file
     * @throws IOException
     */
    public void learnFile(File file) {
        // ����������,ֻ���״�ѵ��ʱ���г�ʼ��
        // ����ѵ��ʱֻ�ܼ���ѵ��ԭ�еĴʻ㣬�޷�ѵ���´�
        if (wordMap.size() == 0) {
            // ��Ƶͳ��
            Map<String, Integer> countMap = new HashMap<String, Integer>();
            readVocab(file, countMap);
            for (Entry<String, Integer> element : countMap.entrySet()) {
                // ��Ƶ̫С�Ĵʲ����м���
                if (element.getValue() < freqThresold) {
                    continue;
                }
                // ��ʼ��������
                wordMap.put(element.getKey(), new WordNeuron(element.getKey(), element.getValue(), layerSize));
            }
            // ����haffman����ʵ������߹�����ɺ�ÿ���������ʹ���һ��ͨ��haffman�����ڵ��·��
            haffman = new Haffman(layerSize);
            haffman.make(wordMap.values());
            System.out.println("haffman����ʼ������");
            // ����ÿ����Ԫ·��������¼�ֲ��¼
            for (Neuron neuron : wordMap.values()) {
                ((WordNeuron) neuron).makeNeurons();
            }
        }
        // ����ѵ��
        for (int i = 0; i < iteratorNum; i++) {
            trainModel(file);
            System.out.println("��" + (i + 1) + "�ε�������");
        }
        System.out.println("����������: " + wordMap.size());
        System.out.println("ѵ���ܴ���: " + trainWordsCount);
    }

    /**
     * ͳ�ƴ�Ƶ
     * 
     * @param file
     * @throws IOException
     */
    private void readVocab(File file, Map<String, Integer> countMap) {
        if (file.isFile()) {
            readVocabFromFile(file, countMap);
        } else if (file.isDirectory()) {
            File[] listFiles = file.listFiles();
            for (File listFile : listFiles) {
                readVocab(listFile, countMap);
            }
        }
    }

    /**
     * �ӵ����ļ���ͳ�ƴ�Ƶ
     * 
     * @param file
     * @param countMap
     */
    private void readVocabFromFile(File file, Map<String, Integer> countMap) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
            String temp = null;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split("\\s+");
                trainWordsCount += split.length;
                for (String string : split) {
                    if (countMap.containsKey(string)) {
                        countMap.put(string, countMap.get(string) + 1);
                    } else {
                        countMap.put(string, 1);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println("��ȡ" + file.getName() + "�ʻ����");
        System.out.println("�ʵ䳤�ȣ�" + countMap.size());
    }

    /**
     * trainModel ֧��ѵ������Ϊһ�������ļ�
     * 
     * @throws IOException
     */
    private void trainModel(File file) {
        if (file.isFile()) {
            trainModelFromFile(file);
        } else if (file.isDirectory()) {
            File[] listFiles = file.listFiles();
            for (File listFile : listFiles) {
                trainModel(listFile);
            }
        }
    }

    private void trainModelFromFile(File file) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
            String temp = null;
            long nextRandom = 5;
            int wordCount = 0;
            int lastWordCount = 0;
            while ((temp = br.readLine()) != null) {
                // ѧϰ�����𲽼�С����ֹ������Ծ
                if (wordCount - lastWordCount > 10000) {
                    System.out.println("alpha:" + alpha + " Progress: "
                            + (int) (wordCountActual / ((double) trainWordsCount * iteratorNum + 1) * 100) + "%");
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / ((double) trainWordsCount * iteratorNum + 1));
                    // ��ֹalpha��С
                    if (alpha < startingAlpha * 0.0001) {
                        alpha = startingAlpha * 0.0001;
                    }
                }
                // ������Ҫ�ÿո����
                String[] strs = temp.split("\\s+");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; i++) {
                    Neuron entry = wordMap.get(strs[i]);
                    if (entry == null) {
                        continue;
                    }

                    // ���ڴ�Ƶ��Ĵʣ����������ѵ������
                    if (sample > 0) {
                        double ran = (Math.sqrt(entry.getFreq() / (sample * trainWordsCount)) + 1)
                                * (sample * trainWordsCount) / entry.getFreq();
                        nextRandom = nextRandom * 25214903917L + 11;
                        if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                            continue;
                        }
                    }
                    sentence.add((WordNeuron) entry);
                }

                for (int index = 0; index < sentence.size(); index++) {
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (isCbow) {
                        cbowGram(index, sentence, (int) nextRandom % window);
                    } else {
                        skipGram(index, sentence, (int) nextRandom % window);
                    }
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println("ѵ��" + file.getName() + "����������");
    }

    /**
     * skip gram ģ��ѵ��
     * 
     * @param sentence
     * @param neu1
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) {
                continue;
            }

            double[] neu1e = new double[layerSize];// �����
            // HIERARCHICAL SOFTMAX
            List<Neuron> neurons = word.getNeurons();
            WordNeuron we = sentence.get(c);
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.getSyn0()[j] * out.getSyn1()[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    // ��fֵ�ܴ���ߺ�Сʱlogisticsֵ����0����1
                    continue;
                } else {
                    // ����ȡlogistics����ֵ
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }

                // ��ʱ��fֵΪlogistics�����������ֵ
                // 'g' is the gradient multiplied by the learning rate
                double g = f * (1 - f) * (1 - word.getCodeArr()[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    // ͳ�����
                    neu1e[c] += g * out.getSyn1()[c];
                }
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.getSyn1()[c] += g * we.getSyn0()[c];
                }
            }

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.getSyn0()[j] += neu1e[j];
            }
        }

    }

    /**
     * �ʴ�ģ��
     * 
     * @param index
     * @param sentence
     * @param b
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;
        // ��ǰ��·��
        List<Neuron> neurons = word.getNeurons();
        // �����
        double[] neu1e = new double[layerSize];
        // �����Ĵ�������
        double[] neu1 = new double[layerSize];
        WordNeuron lastWord;

        // ѭ�������ڴʻ㣬�����������
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) {
                    continue;
                }
                // �����ĵĴ�
                lastWord = sentence.get(c);
                if (lastWord == null) {
                    continue;
                }
                // �����Ĵ��������
                for (c = 0; c < layerSize; c++) {
                    neu1[c] += lastWord.getSyn0()[c];
                }
            }
        }
        // HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            // ��������ȡ������Ԫ�ڵ�
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            for (c = 0; c < layerSize; c++) {
                f += neu1[c] * out.getSyn1()[c];
            }
            // f�ܴ���Сʱ��logisticsֵ����0����1
            if (f <= -MAX_EXP) {
                f = 0;
            } else if (f >= MAX_EXP) {
                f = 1;
            } else {
                // ���
                // x=(y/6+1)*1000/2
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            }

            // 'g' is the gradient multiplied by the learning rate
            // double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = (word.getCodeArr()[d] - f) * alpha;

            for (c = 0; c < layerSize; c++) {
                // �ۼ������
                neu1e[c] += g * out.getSyn1()[c];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                // ��������֦�ɽڵ�����
                out.getSyn1()[c] += g * neu1[c];
            }
        }
        // ���ۼ����������������Ĵ��ڵ�ÿ����������
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) {
                    continue;
                }
                lastWord = sentence.get(c);
                if (lastWord == null) {
                    continue;
                }
                for (c = 0; c < layerSize; c++)
                    lastWord.getSyn0()[c] += neu1e[c];
            }
        }
    }

    /**
     * ����ģ�ͣ�ֻ���������
     */
    public void saveWordVecs(File file) {
        DataOutputStream dataOutputStream = null;
        try {
            dataOutputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
            // ����������
            dataOutputStream.writeInt(wordMap.size());
            // ����������
            dataOutputStream.writeInt(layerSize);
            float[] syn0 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron) element.getValue()).getSyn0();
                for (float d : syn0) {
                    dataOutputStream.writeFloat(d);
                }
            }
        } catch (Exception e) {
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
}
