package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;


public class Example {

	public static void main(String[] args) {
		String arffFileName = "ARFF/MatchingNetwork-online_book-COMA-Matcher.arff";
		BufferedReader reader;
		Instances allInstances,trainingInstances,testInstances;
		allInstances = null;
		try {
			reader = new BufferedReader(new FileReader(new File(arffFileName)));
			allInstances = new Instances(reader);
			reader.close();
		} catch (FileNotFoundException e) {
			System.err.printf("ARFF file not found.\n");
		} catch (IOException e) {
			System.err.printf("Error reading ARFF file.\n");
		}
		
		allInstances.deleteAttributeAt(1);				//deleting candidate string attribute
		allInstances.setClassIndex(allInstances.numAttributes()-1);
		Random r = null;
		for (int randSeed = 0; randSeed < 50; randSeed++){
		r = new Random(randSeed);
		allInstances.randomize(r);
		int POSITIVE = allInstances.classAttribute().indexOfValue("true");
		int NEGATIVE = 1 - POSITIVE;
		String[] classes = new String[2]; 
		classes[POSITIVE] = "TRUE";
		classes[NEGATIVE] = "FALSE";
//		System.out.printf("Positive index is %d\n",POSITIVE);
//		System.out.printf("---------------------------------------------\n");
		try {
			RandomForest rf = new RandomForest();
			int div = 0;
			for (int i=0; i<5; i++){
//			int i=3;
				Evaluation eval = new Evaluation(allInstances);
				trainingInstances = allInstances.trainCV(5, i);
				testInstances = allInstances.testCV(5, i);
				rf.buildClassifier(trainingInstances);
				
				eval.evaluateModel(rf, testInstances);
//				System.out.printf("%s\n",eval.toSummaryString());
				int tp = (int)eval.numTruePositives(POSITIVE);
				int fp = (int)eval.numFalsePositives(POSITIVE);
				int fn = (int)eval.numFalseNegatives(POSITIVE);
//				System.out.printf("Calc: TP: %d \tFP: %d\tPrec: %.3f\n",tp,fp,tp/(double)(tp+fp));
//				System.out.printf("Calc: FN: %d \tRec: %.3f\n",fn,tp/(double)(tp+fn));
//				System.out.printf("TP-Rate: %.3f\n",eval.truePositiveRate(POSITIVE));
//				System.out.printf("Precision: %.3f\n",eval.precision(POSITIVE));
//				System.out.printf("Recall: %.3f\n",eval.recall(POSITIVE));
//				System.out.printf("F-Measure: %.3f\n",eval.fMeasure(POSITIVE));
//				System.out.printf("^--------------- FOLD NUMBER %d ---------------^\n",i+1);
				Classifier[] trees = rf.getClassifiers();
				int[] countAgree = new int[10];
				for (int instID = 0; instID < testInstances.numInstances(); instID++){
					Instance inst = testInstances.instance(instID);
					int real = (int)inst.classValue();
					int pred = (int)rf.classifyInstance(inst);
					double[] probs = rf.distributionForInstance(inst);
					if ((probs[0] >= 0.5) && (probs[0] <= 0.5)) div++;
					/*if ((pred == POSITIVE)||(real == POSITIVE)){
						System.out.printf("Prediction: %5s | Real: %5s |\t" +
								"Prob(T): %.3f \t | Prob(F): %.3f\t",classes[pred],classes[real],probs[POSITIVE],probs[NEGATIVE]);
						for (int tID = 0; tID < 10; tID++)
							System.out.printf("%s ",trees[tID].classifyInstance(inst)==POSITIVE?"T":"F");
						System.out.printf("\t|\t");
						for (int tID = 0; tID < 10; tID++){
							if (trees[tID].classifyInstance(inst) == pred){
								System.out.printf("X ");
								countAgree[tID]++;
							}else System.out.printf("  ");
						}
						System.out.printf("\n--------------------------\n");
					}*/
				}
				//for (int tID = 0; tID < 10; tID++)
				//	System.out.printf("Tree %d agreed %d times with the forest\n",tID,countAgree[tID]);
			}
			System.out.printf("Dead locks: %d out of %d\n",div,allInstances.numInstances());
		} catch (Exception e) {
			e.printStackTrace();
		}
		}
	}

}
