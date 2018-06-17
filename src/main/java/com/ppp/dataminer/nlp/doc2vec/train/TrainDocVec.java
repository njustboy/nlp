package com.ppp.dataminer.nlp.doc2vec.train;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.ppp.dataminer.nlp.doc2vec.data.HiddenNeuron;
import com.ppp.dataminer.nlp.doc2vec.data.Neuron;
import com.ppp.dataminer.nlp.doc2vec.data.WordNeuron;

/**
 * 文本向量训练工具
 * 
 * 注意需要提前训练好词向量及haffman树模型，训练文本向量时词向量不会改变
 * @author zhangwei
 *
 */
public class TrainDocVec extends TrainVec {
    /**
     * 文本向量
     */
    private Map<String, float[]> docVector = new HashMap<>();

    /**
     * 从内存构建模型
     * 
     * @param wordMap
     * @throws IOException
     */
    public TrainDocVec(Map<String, Neuron> wordMap) {
        super(wordMap);
    }

    /**
     * 从模型文件构建模型
     * 
     * @param modelFile
     */
    public TrainDocVec(File modelFile) {
        super(modelFile);
    }

    public TrainDocVec() {
        super();
    }

    /**
     * 预测文本向量
     * 
     * @param words
     * @return
     */
    public float[] calcVector(String[] words) {
        float[] vector = new float[layerSize];
        // 初始化文本向量
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) ((new Random().nextDouble() - 0.5) / layerSize);
        }
        // 迭代计算
        for (int j = 0; j < iteratorNum; j++) {
            alpha = startingAlpha * (1 - (double) j / iteratorNum);
            long nextRandom = new Random().nextInt(5);
            List<WordNeuron> sentence = new ArrayList<WordNeuron>();
            for (int i = 0; i < words.length; i++) {
                Neuron entry = wordMap.get(words[i]);
                if (entry == null) {
                    continue;
                }
                // 随机跳过一些词汇，只有出现频率很大的词在这里才可能被跳过
                // if (sample > 0) {
                // double ran = (Math.sqrt(entry.freq / (sample *
                // trainWordsCount))
                // + 1) * (sample * trainWordsCount)
                // / entry.freq;
                // nextRandom = nextRandom * 25214903917L + 11;
                // if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                // continue;
                // }
                // }
                sentence.add((WordNeuron) entry);
            }

            for (int index = 0; index < sentence.size(); index++) {
                nextRandom = nextRandom * 25214903917L + 11;
                if (isCbow) {
                    cbowGram(index, vector, sentence, (int) nextRandom % window);
                } else {
                    skipGram(index, vector, sentence, (int) nextRandom % window);
                }
            }
        }

        return vector;
    }

    /**
     * 根据文件学习，批量应用
     * 
     * @param file
     * @throws IOException
     */
    public void learnFile(File file) {
        // 初始化文本向量
        initializeDocVec(file);

        for (int i = 0; i < iteratorNum; i++) {
            // 每一轮迭代减小学习步长
            alpha = startingAlpha * (1 - (double) i / iteratorNum);
            trainModel(file);
        }
    }

    /**
     * trainModel
     * 
     * @param file
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
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String temp = null;
            long nextRandom = 5;
            int wordCount = 0;
            int lastWordCount = 0;
            int sentNo = 0;
            String fileName = file.getName();
            while ((temp = br.readLine()) != null) {
                if (wordCount - lastWordCount > 10000) {
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    // alpha = startingAlpha * (1 - wordCountActual / ((double)
                    // trainWordsCount * iteratorNum + 1));
                    // if (alpha < startingAlpha * 0.0001) {
                    // alpha = startingAlpha * 0.0001;
                    // }
                }
                String[] strs = temp.split("\\s+");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; i++) {
                    Neuron entry = wordMap.get(strs[i]);
                    if (entry == null) {
                        continue;
                    }
                    // The subsampling randomly discards frequent words while
                    // keeping the ranking same
                    // if (sample > 0) {
                    // double ran = (Math.sqrt(entry.freq / (sample *
                    // trainWordsCount)) + 1) * (sample * trainWordsCount)
                    // / entry.freq;
                    // nextRandom = nextRandom * 25214903917L + 11;
                    // if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                    // continue;
                    // }
                    // }
                    sentence.add((WordNeuron) entry);
                }

                for (int index = 0; index < sentence.size(); index++) {
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (isCbow) {
                        cbowGram(index, fileName + "_" + sentNo, sentence, (int) nextRandom % window);
                    } else {
                        skipGram(index, fileName + "_" + sentNo, sentence, (int) nextRandom % window);
                    }
                }
                sentNo++;
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
        System.out.println("训练" + file.getName() + "文本向量结束");
    }

    /**
     * skip gram 模型训练
     * 
     * 模型未验证，暂时不用
     * 
     * @param sentence
     * @param neu1
     */
    private void skipGram(int index, String key, List<WordNeuron> sentence, int b) {
        float[] docVec = docVector.get(key);
        skipGram(index, docVec, sentence, b);
    }

    private void skipGram(int index, float[] docVec, List<WordNeuron> sentence, int b) {
        // WordNeuron word = sentence.get(index);
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
            WordNeuron we = sentence.get(c);
            List<Neuron> neurons = we.getNeurons();

            // 不是中间词向量，而是文本向量

            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    // f += we.syn0[j] * out.syn1[j];
                    f += docVec[j] * out.getSyn1()[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // 'g' is the gradient multiplied by the learning rate
                double g = (1 - we.getCodeArr()[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    neu1e[c] += g * out.getSyn1()[c];
                }
                // Learn weights hidden -> output
                // for (c = 0; c < layerSize; c++) {
                // out.syn1[c] += g * we.syn0[c];
                //
                // }
                // 不改变预测的中间词的向量
            }

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                // we.syn0[j] += neu1e[j];

                docVec[j] += neu1e[j];
                // 更新句子（文本）向量，不更新词向量
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
    private void cbowGram(int index, String key, List<WordNeuron> sentence, int b) {
        float[] docVec = docVector.get(key);
        cbowGram(index, docVec, sentence, b);
    }

    /**
     * 词袋模型
     * 
     * @param index
     * @param sentence
     * @param b
     */
    private void cbowGram(int index, float[] vector, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;

        // haffman树路径
        List<Neuron> neurons = word.getNeurons();
        // 误差项
        double[] neu1e = new double[layerSize];
        // 上下文词向量和
        double[] neu1 = new double[layerSize];
        WordNeuron lastWord;
        // 此处不是取固定窗口，而是取最大窗口大小以内的一个随机值
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
                for (c = 0; c < layerSize; c++) {
                    // 计算上下文词向量和
                    neu1[c] += lastWord.getSyn0()[c];
                }
            }
        }
        // 将文本的向量也作为输入
        for (c = 0; c < layerSize; c++) {
            neu1[c] += vector[c];
        }

        // 层次SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++)
                f += neu1[c] * out.getSyn1()[c];
            if (f <= -MAX_EXP || f >= MAX_EXP) {
                continue;
            } else {
                // 差静态表获得结果（减少计算量）
                // y=ax+b
                // 1000=6a+b
                // 0=-6a+b
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            }
            // 'g' is the gradient multiplied by the learning rate
            // double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = f * (1 - f) * (word.getCodeArr()[d] - f) * alpha;
            // 累加误差
            for (c = 0; c < layerSize; c++) {
                neu1e[c] += g * out.getSyn1()[c];
            }

        }
        // 更新文本向量
        for (c = 0; c < layerSize; c++) {
            vector[c] += neu1e[c];
        }
    }

    /**
     * 初始化文本向量
     * 
     * @param file
     * @throws IOException
     */
    private void initializeDocVec(File file) {
        if (file.isFile()) {
            initializeDocVecFromFile(file);
        } else if (file.isDirectory()) {
            File[] listFiles = file.listFiles();
            for (File listFile : listFiles) {
                initializeDocVec(listFile);
            }
        }

    }

    /**
     * 初始化文本向量，输入文本中的每一行表示一个文本，需要预先进行分词处理并将分词结果空格隔开
     * @param file
     */
    private void initializeDocVecFromFile(File file) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String temp = null;
            int sent_no = 0;
            String fileName = file.getName();
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split("\\s+");
                trainWordsCount += split.length;
                float[] vector = new float[layerSize];

                Random random = new Random();

                for (int i = 0; i < vector.length; i++)
                    vector[i] = (float) ((random.nextDouble() - 0.5) / layerSize);

                docVector.put(fileName + "_" + sent_no, vector);

                sent_no++;
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
        System.out.println("初始化" + file.getName() + "文本向量结束");
    }

    /**
     * 文本向量写文件
     * 
     * @param file
     */
    public void saveDocVecs(File file, int maxCount) {
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(file));
            StringBuilder sb = new StringBuilder();
            Object[] array = docVector.keySet().toArray();
            Arrays.sort(array);
            String head = ((String) array[0]).substring(0, ((String) array[0]).lastIndexOf("_"));
            // 计数器
            int count = 0;
            for (Object docNo : array) {
                count++;
                sb.delete(0, sb.length());
                String thisHead = ((String) docNo).substring(0, ((String) docNo).lastIndexOf("_"));
                if (!head.equals(thisHead)) {
                    head = thisHead;
                    count = 1;
                } else if (count > maxCount) {
                    continue;
                }
                sb.append(thisHead + " ");
                float[] vector = docVector.get(docNo);
                float sum = 0f;
                for (float f : vector) {
                    sum += f * f;
                }
                sum = (float) Math.sqrt(sum);
                sum = sum == 0 ? 1 : sum;
                for (float e : vector) {
                    sb.append(e / sum + " ");
                }
                bw.write(sb.toString().trim());
                bw.newLine();
            }
            bw.flush();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
