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
 * 词向量训练工具 语料需要先用分词工具分词，词汇间用空格隔开
 * 
 * @author zhangwei
 *
 */
public class TrainWordVec extends TrainVec {
    /**
     * 从模型文件加载模型
     * 
     * @param modelFile
     */
    public TrainWordVec(File modelFile) {
        super(modelFile);
    }

    /**
     * 从内存加载模型
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
     * 根据文件学习
     * 
     * @param file
     * @throws IOException
     */
    public void learnFile(File file) {
        // 构建词向量,只有首次训练时进行初始化
        // 增量训练时只能继续训练原有的词汇，无法训练新词
        if (wordMap.size() == 0) {
            // 词频统计
            Map<String, Integer> countMap = new HashMap<String, Integer>();
            readVocab(file, countMap);
            for (Entry<String, Integer> element : countMap.entrySet()) {
                // 词频太小的词不进行计算
                if (element.getValue() < freqThresold) {
                    continue;
                }
                // 初始化词向量
                wordMap.put(element.getKey(), new WordNeuron(element.getKey(), element.getValue(), layerSize));
            }
            // 构建haffman树，实际上这边构建完成后每个词向量就存在一条通向haffman树根节点的路径
            haffman = new Haffman(layerSize);
            haffman.make(wordMap.values());
            System.out.println("haffman树初始化结束");
            // 查找每个神经元路径，并记录分叉记录
            for (Neuron neuron : wordMap.values()) {
                ((WordNeuron) neuron).makeNeurons();
            }
        }
        // 迭代训练
        for (int i = 0; i < iteratorNum; i++) {
            trainModel(file);
            System.out.println("第" + (i + 1) + "次迭代结束");
        }
        System.out.println("词向量总数: " + wordMap.size());
        System.out.println("训练总词数: " + trainWordsCount);
    }

    /**
     * 统计词频
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
     * 从单个文件中统计词频
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
        System.out.println("读取" + file.getName() + "词汇结束");
        System.out.println("词典长度：" + countMap.size());
    }

    /**
     * trainModel 支持训练语料为一个或多个文件
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
                // 学习速率逐步减小，防止出现跳跃
                if (wordCount - lastWordCount > 10000) {
                    System.out.println("alpha:" + alpha + " Progress: "
                            + (int) (wordCountActual / ((double) trainWordsCount * iteratorNum + 1) * 100) + "%");
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / ((double) trainWordsCount * iteratorNum + 1));
                    // 防止alpha过小
                    if (alpha < startingAlpha * 0.0001) {
                        alpha = startingAlpha * 0.0001;
                    }
                }
                // 语料需要用空格隔开
                String[] strs = temp.split("\\s+");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; i++) {
                    Neuron entry = wordMap.get(strs[i]);
                    if (entry == null) {
                        continue;
                    }

                    // 对于词频大的词，随机减少其训练概率
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
        System.out.println("训练" + file.getName() + "词向量结束");
    }

    /**
     * skip gram 模型训练
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

            double[] neu1e = new double[layerSize];// 误差项
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
                    // 当f值很大或者很小时logistics值趋于0或者1
                    continue;
                } else {
                    // 查表获取logistics近似值
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }

                // 此时的f值为logistics函数算出来的值
                // 'g' is the gradient multiplied by the learning rate
                double g = f * (1 - f) * (1 - word.getCodeArr()[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    // 统计误差
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
     * 词袋模型
     * 
     * @param index
     * @param sentence
     * @param b
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;
        // 当前词路径
        List<Neuron> neurons = word.getNeurons();
        // 误差项
        double[] neu1e = new double[layerSize];
        // 上下文词向量和
        double[] neu1 = new double[layerSize];
        WordNeuron lastWord;

        // 循环窗口内词汇，将词向量相加
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) {
                    continue;
                }
                // 上下文的词
                lastWord = sentence.get(c);
                if (lastWord == null) {
                    continue;
                }
                // 上下文词向量相加
                for (c = 0; c < layerSize; c++) {
                    neu1[c] += lastWord.getSyn0()[c];
                }
            }
        }
        // HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            // 从上往下取隐层神经元节点
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            for (c = 0; c < layerSize; c++) {
                f += neu1[c] * out.getSyn1()[c];
            }
            // f很大或很小时，logistics值趋于0或者1
            if (f <= -MAX_EXP) {
                f = 0;
            } else if (f >= MAX_EXP) {
                f = 1;
            } else {
                // 查表
                // x=(y/6+1)*1000/2
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            }

            // 'g' is the gradient multiplied by the learning rate
            // double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = (word.getCodeArr()[d] - f) * alpha;

            for (c = 0; c < layerSize; c++) {
                // 累加误差项
                neu1e[c] += g * out.getSyn1()[c];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                // 迭代更新枝干节点向量
                out.getSyn1()[c] += g * neu1[c];
            }
        }
        // 将累加误差项更新至上下文窗口的每个词向量中
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
     * 保存模型，只保存词向量
     */
    public void saveWordVecs(File file) {
        DataOutputStream dataOutputStream = null;
        try {
            dataOutputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
            // 词向量数量
            dataOutputStream.writeInt(wordMap.size());
            // 词向量长度
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
