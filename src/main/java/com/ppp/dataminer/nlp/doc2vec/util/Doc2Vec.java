package com.ppp.dataminer.nlp.doc2vec.util;

import java.io.File;

import com.ppp.dataminer.nlp.doc2vec.train.TrainDocVec;

public class Doc2Vec {
    private TrainDocVec trainDocVec = null;

    private volatile static Doc2Vec instance = null;

    private Doc2Vec() {
        trainDocVec = new TrainDocVec(new File("model/haffman_15_100.mod"));
    }

    public static Doc2Vec getInstance() {
        if (instance == null) {
            synchronized (Doc2Vec.class) {
                instance = new Doc2Vec();
            }
        }
        return instance;
    }

    public float[] calcDocVec(String[] words) {
        float[] calcVector = trainDocVec.calcVector(words);
        convertVec(calcVector);
        return calcVector;
    }

    private static void convertVec(float[] vector) {
        float sum = 0;
        for (float f : vector) {
            sum += f * f;
        }
        sum = (float) Math.sqrt(sum);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= sum;
        }
    }

    public float getDistance(float[] vector1, float[] vector2) {
        float distance = 0;
        for (int i = 0; i < vector1.length; i++) {
            distance += vector1[i] * vector2[i];
        }
        return distance;
    }
    
    public float getDistance(String[] words1,String[] words2){
        float[] vector1 = calcDocVec(words1);
        float[] vector2 = calcDocVec(words2);
        
        return getDistance(vector1,vector2);
    }

}
