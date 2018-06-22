package com.ppp.dataminer.nlp.doc2vec.data;

public class WordPair implements Comparable<WordPair> {

    private String word;
    private double weight;

    public WordPair() {

    }

    public WordPair(String word, double weight) {
       this.word = word;
       this.weight = weight;
    }

    public String getWord() {
       return word;
    }

    public double getWeight() {
       return weight;
    }

    public void setWord(String word) {
       this.word = word;
    }

    public void setWeight(double weight) {
       this.weight = weight;
    }

    public int compareTo(WordPair o) {
       if (weight > o.getWeight()) {
          return -1;
       }
       if (weight < o.getWeight()) {
          return 1;
       }
       return 0;
    }

    public String toString() {
//       return word + ":" + weight;
    	return word;
    }

    public boolean equals(Object obj) {
       //
       if (this == obj) {
          return true;
       }
       if (null == obj) {
          return false;
       }
       if (getClass() != obj.getClass()) {
          return false;
       }
       WordPair o = (WordPair) obj;
       if (!word.equals(o.word)) {
          return false;
       }
       if (weight != o.weight) {
          return false;
       }
       return true;
    }

}
