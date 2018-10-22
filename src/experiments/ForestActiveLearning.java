package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;


public class ForestActiveLearning {

	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
//		domain = "betting";
//		domain = "business_partner";
//		domain = "magazine_subscription";
		domain = "online_book";
//		domain = "purchase_order";
		
		double MIN, MAX;
//		MIN = 0.4; MAX = 0.6;
		MIN = MAX = 0.5;
		String arffFileName = "ARFF/MatchingNetwork-" + domain + "-COMA-Matcher.arff";
		double trainingPerc = 10;
		BufferedReader reader;
		Instances allInstances,trainingInstances,testInstances;
		PrintWriter pw = null;
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
//		for (int randSeed = 0; randSeed < 50; randSeed++){
		for (int randSeed = 0; randSeed < 30; randSeed++){
			String logFileName = "logs/ForestActiveLearning/"+ domain + "/" + domain + "-" + randSeed + ".log.txt";
			try {
				pw = new PrintWriter(new File(logFileName));
			} catch (FileNotFoundException e1) {
				System.err.printf("Error writing log (%s)\n",logFileName);
			}
			System.out.printf("Running %s (seed %d)\t",domain,randSeed);
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			r = new Random(randSeed);
			allInstances.randomize(r);
			int POSITIVE = allInstances.classAttribute().indexOfValue("true");
			int NEGATIVE = 1 - POSITIVE;
			String[] classes = new String[2]; 
			classes[POSITIVE] = "TRUE";
			classes[NEGATIVE] = "FALSE";
			//System.out.printf("Positive index is %d\n",POSITIVE);
			
			//Choosing training instances
			int numInstances = allInstances.numInstances();
			double percTraining = trainingPerc/100.0;
			int maxTrainingInstances = (int)(numInstances * percTraining);
			trainingInstances.delete();
			
			List<Integer> trainingList = new ArrayList<>();
			for (int i = 0; i < numInstances; i++)
				trainingList.add(i);
			Collections.shuffle(trainingList,r);
			trainingList = trainingList.subList(0, maxTrainingInstances);
			Collections.sort(trainingList);
			for (int i = maxTrainingInstances-1; i >= 0; i--){
				int instanceID = trainingList.get(i);
				Instance inst = testInstances.instance(instanceID);
				testInstances.delete(instanceID);
				trainingInstances.add(inst);
			}
			
			//System.out.printf("Training: %d\nTest: %d\n",trainingInstances.numInstances(),testInstances.numInstances());
//			System.out.printf("#Training\tTP\tFP\tFN\tPrec\tRec\tF1Score\t#Deadlocks\n");
			pw.printf("#Training\tTP\tFP\tFN\tPrec\tRec\tF1Score\t#Deadlocks\n");
			
			try {
				int rounds = 1;
				int deadlocks = 0;
				do{
					deadlocks = 0;
					//System.out.printf("Training.size: %d\n",trainingInstances.numInstances());
					RandomForest rf = new RandomForest();
					Evaluation eval = new Evaluation(allInstances);
					
					rf.buildClassifier(trainingInstances);
					eval.evaluateModel(rf, testInstances);
		
					//System.out.printf("%s\n",eval.toSummaryString());
					int tp = (int)eval.numTruePositives(POSITIVE);
					int fp = (int)eval.numFalsePositives(POSITIVE);
					int fn = (int)eval.numFalseNegatives(POSITIVE);
					//System.out.printf("Calc: TP: %d \tFP: %d\tPrec: %.3f\n",tp,fp,tp/(double)(tp+fp));
					//System.out.printf("Calc: FN: %d \tRec: %.3f\n",fn,tp/(double)(tp+fn));
					//System.out.printf("TP-Rate: %.3f\n",eval.truePositiveRate(POSITIVE));
					double prec = eval.precision(POSITIVE);
					//System.out.printf("Precision: %.3f\n",eval.precision(POSITIVE));
					double rec = eval.recall(POSITIVE);
					//System.out.printf("Recall: %.3f\n",eval.recall(POSITIVE));
					double f1score = eval.fMeasure(POSITIVE);
					//System.out.printf("F-Measure: %.3f\n",eval.fMeasure(POSITIVE));
					//System.out.printf("^--------------- FOLD NUMBER %d ---------------^\n",i+1);
					//Classifier[] trees = rf.getClassifiers();
					//int[] countAgree = new int[10];
					List<Integer> deadlockInstances = new ArrayList<Integer>();
					for (int instID = 0; instID < testInstances.numInstances(); instID++){
						Instance inst = testInstances.instance(instID);
						//int real = (int)inst.classValue();
						//int pred = (int)rf.classifyInstance(inst);
						double[] probs = rf.distributionForInstance(inst);
						if ((probs[0] >= MIN) && (probs[0] <= MAX)){
							deadlocks++;
							deadlockInstances.add(instID);
							//System.out.printf("instance %d no decision\n",instID);
						}
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
					Collections.sort(deadlockInstances);
					for (int i = deadlocks-1; i >= 0; i--){
						int instanceID = deadlockInstances.get(i);
						Instance inst = testInstances.instance(instanceID);
						testInstances.delete(instanceID);
						trainingInstances.add(inst);
						//System.out.printf("Instance %d added to training\n",instanceID);
					}
					//for (int tID = 0; tID < 10; tID++)
					//	System.out.printf("Tree %d agreed %d times with the forest\n",tID,countAgree[tID]);
					//System.out.printf("Dead locks: %d out of %d\n",deadlocks,allInstances.numInstances());
//					System.out.printf("%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%d\n",trainingInstances.numInstances(),tp,fp,fn,prec,rec,f1score,deadlocks);
					pw.printf("%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%d\n",trainingInstances.numInstances(),tp,fp,fn,prec,rec,f1score,deadlocks);
					System.out.printf("%d ",rounds);
					rounds++;
					//System.out.printf("----------------------------------------------\n\n");
				}while ((deadlocks > 0) && (testInstances.numInstances() > 0));
			} catch (Exception e) {
				e.printStackTrace();
			}
			pw.close();
			System.out.printf("Done.\n");
		}
	}

}
