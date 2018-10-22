package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Experiment performed to measure how many times a deadlock (split decision by forest of trees) occur 
 * in the first iteration of the execution.
 * @author diego
 *
 */
public class NumberOfDeadlocks {

	public static void main(String[] args) {
//		String arffFileName = "ARFF/MatchingNetwork-betting-COMA-Matcher.arff";
//		String arffFileName = "ARFF/MatchingNetwork-business_partner-COMA-Matcher.arff";
//		String arffFileName = "ARFF/MatchingNetwork-magazine_subscription-COMA-Matcher.arff";
//		String arffFileName = "ARFF/MatchingNetwork-online_book-COMA-Matcher.arff";
		String arffFileName = "ARFF/MatchingNetwork-purchase_order-COMA-Matcher.arff";
		double MIN, MAX;
//		MIN = 0.4; MAX = 0.6;
		MIN = MAX = 0.5;
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
		int sumDeadlocks05 = 0;
		int[] history0406 = new int[30];
		for (int randSeed = 0; randSeed < 30; randSeed++){
			r = new Random(randSeed);
			allInstances.randomize(r);
			try {
				RandomForest rf = new RandomForest();
				int div05 = 0;
				int div0406 = 0;
				for (int i=0; i<5; i++){
					Evaluation eval = new Evaluation(allInstances);
					trainingInstances = allInstances.trainCV(5, i);
					testInstances = allInstances.testCV(5, i);
					rf.buildClassifier(trainingInstances);

					eval.evaluateModel(rf, testInstances);
					for (int instID = 0; instID < testInstances.numInstances(); instID++){
						Instance inst = testInstances.instance(instID);
						double[] probs = rf.distributionForInstance(inst);
						if (probs[0] == 0.5) div05++;
						if ((probs[0] >= 0.4) && (probs[0] <= 0.6)) div0406++;
					}
				}
				sumDeadlocks05 += div05;
				history0406[randSeed] = div0406;
				System.out.printf("Dead locks: %d out of %d (%.3f%%)\n",
						div05,allInstances.numInstances(),(div05*100.0)/(double)allInstances.numInstances());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		System.out.printf("\nDataset: %s\n" +
				"Total dead locks: %d.\n" +
				"Average per run: %.2f\n",arffFileName,sumDeadlocks05,sumDeadlocks05/30.0);
		System.out.printf("Lower bound probability: %.2f\n" +
				"Upper bound probability: %.2f\n",0.5,0.5);
		System.out.printf("-------------------------------------------------------------\n");
		int sumDeadlocks0406 = 0;
		for (int i=0; i<30; i++){
			int div0406 = history0406[i];
			sumDeadlocks0406 += div0406;
			System.out.printf("Dead locks: %d out of %d (%.3f%%)\n",
					div0406,allInstances.numInstances(),(div0406*100.0)/(double)allInstances.numInstances());
		}
		System.out.printf("\nDataset: %s\n" +
				"Total dead locks: %d.\n" +
				"Average per run: %.2f\n",arffFileName,sumDeadlocks0406,sumDeadlocks0406/30.0);
		System.out.printf("Lower bound probability: %.2f\n" +
				"Upper bound probability: %.2f\n",0.4,0.6);
		
	}

}
