package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Vote Classifier test class
 * @author diego
 *
 */
public class VoteClassifier {

	public static void main(String[] args) {
		String arffFileName = "ARFF/MatchingNetwork-online_book-COMA-Matcher.arff";
		double MIN, MAX;
		MIN = MAX = 0.5;
		int NUMBER_OF_TREES = 10;
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
		int POSITIVE = allInstances.classAttribute().indexOfValue("true");
		int NEGATIVE = 1 - POSITIVE;
		
		for (int randSeed = 0; randSeed < 30; randSeed++){
			r = new Random(randSeed);
			allInstances.randomize(r);
			try {
				RandomForest rf = new RandomForest();
				rf.setNumTrees(NUMBER_OF_TREES);
				
				for (int i=0; i<10; i++){
					Evaluation eval = new Evaluation(allInstances);
					trainingInstances = allInstances.trainCV(10, i);
					testInstances = allInstances.testCV(10, i);
					rf.buildClassifier(trainingInstances);
					
					Classifier[] RFTrees = rf.getClassifiers();
					Vote voters10 = new Vote();
					Classifier[] newTrees = new Classifier[3];
					for (int j=0; j<3; j++)
						newTrees[j] = Classifier.makeCopy(RFTrees[j]);
					voters10.setClassifiers(newTrees);
					
					double[] f1scores = new double[NUMBER_OF_TREES];
					for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
						Evaluation ev = new Evaluation(testInstances);
						ev.evaluateModel(RFTrees[treeID], testInstances);
						//System.out.printf("[%d] F-Measure: %.3f\n",treeID,ev.fMeasure(POSITIVE));
						f1scores[treeID] = ev.fMeasure(POSITIVE);
					}
					
					
					System.out.printf("Voters 10: %s\n",voters10.toString());
					System.exit(0);
					System.out.printf("passou\n");
					
					voters10.setClassifiers(RFTrees);
					eval.evaluateModel(rf, testInstances);
					for (int instID = 0; instID < testInstances.numInstances(); instID++){
						Instance inst = testInstances.instance(instID);
						double[] probs = rf.distributionForInstance(inst);
						if (probs[0] == 0.5) System.out.printf("DL!\n");
						//if ((probs[0] >= 0.4) && (probs[0] <= 0.6)) deadlocksRF0406++;
						probs = voters10.distributionForInstance(inst);
						if (probs[0] == 0.5) System.out.printf("Voters DL!\n");
//						probs = voters20.distributionForInstance(inst);
//						if (probs[0] == 0.5) deadLocksVT20++;
//						probs = voters50.distributionForInstance(inst);
//						if (probs[0] == 0.5) deadLocksVT50++;
					}
					System.out.printf("Seed %d | Fold %d |");
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

}
