package com.ppp.dataminer.nlp.topicmodel.lda;

import java.io.File;
import java.io.IOException;

import com.ppp.dataminer.nlp.topicmodel.data.Documents;
import com.ppp.dataminer.nlp.topicmodel.util.FileUtil;

public class LdaGibbsSampling {
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		trainModel();
		inferenceModel();
	}
	
	private static void trainModel() throws IOException{
        String resultPath = "ldamodel";
		
		Documents docSet = new Documents();
		docSet.readDocs("trainsource");
		System.out.println("wordMap size " + docSet.getTermToIndexMap().size());
		FileUtil.mkdir(new File(resultPath));
		LdaModel model = new LdaModel();
		System.out.println("1 Initialize the model ...");
		model.initializeModel(docSet);
		System.out.println("2 Learning and Saving the model ...");
		model.trainModel(docSet);
		System.out.println("3 Output the final model ...");
		model.saveIteratedModel(100, docSet);
		System.out.println("Done!");
	}
	
	private static void inferenceModel(){
		LDAInference ldaInference = new LDAInference();
		ldaInference.initializeModel("ldamodel/");
		float[] topicVec = ldaInference.ldaInference("店员/营业员#&#&#在面料市场做营业员，每天像顾客推荐布料、款式，让他们能够购买商品。");
		float sum = 0;
		float max = 0;
		int index = 0;
		int maxIndex = 0;
		for(float topicPro:topicVec){
			index++;
			sum += topicPro;
			if(topicPro>max){
				max = topicPro;
				maxIndex = index;
			}
			System.out.print(topicPro+" ");
		}
		System.out.println();
		System.out.println(sum);
		System.out.println(max);
		System.out.println(maxIndex);
	}
}
