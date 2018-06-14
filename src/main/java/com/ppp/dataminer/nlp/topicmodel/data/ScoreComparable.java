package com.ppp.dataminer.nlp.topicmodel.data;

import java.util.Comparator;

public class ScoreComparable implements Comparator<Integer> {
	public float[] sortProb; 

	public ScoreComparable(float[] sortProb) {
		this.sortProb = sortProb;
	}

	@Override
	public int compare(Integer o1, Integer o2) {
		if (sortProb[o1] > sortProb[o2])
			return -1;
		else if (sortProb[o1] < sortProb[o2])
			return 1;
		else
			return 0;
	}

}
