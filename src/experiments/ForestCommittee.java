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

import experiments.functions.Functions;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;


public class ForestCommittee {

	static int COMMITTEE_SIZE = 10;
	static int NUMBER_OF_TREES = 100;
	static double MIN = 0.5;
	static double MAX = 0.5;
	static final String[] domainList = {"betting","business_partner","magazine_subscription","online_book","purchase_order"};
	static String EXPERIMENTS = "";
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		Random r = null;
		int DOMAIN_ID = 4;
		if (args.length > 0){
			EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n" +
					"> Committee Size (10, 20, 50)\n" +
					"> Upper and lower bound of disagreement (default: 0.5 0.5)\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			COMMITTEE_SIZE = Integer.valueOf(args[1]);
			MIN = Double.valueOf(args[2]);
			MAX = Double.valueOf(args[3]);
		}
		domain = domainList[DOMAIN_ID];
		String arffFileName = EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + "-COMA-Matcher.arff";
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
//		for (int randSeed = 0; randSeed < 50; randSeed++){
		for (int randSeed = 0; randSeed < 30; randSeed++){
			String logFileName = EXPERIMENTS + "logs/ForestCommittee" + COMMITTEE_SIZE + "/"+ domain + "/" + domain + "-" + randSeed + ".log.txt";
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
			pw.printf("#Training\tTP\tFP\tFN\tPrec\tRec\tF1Score\tF2Score\t#Deadlocks\t|\tRandomForest\tTP\tFP\tFN\tPrec\tRec\tF1Score\n");
			
			try {
				int rounds = 1;
				int deadlocks = 0;
				do{
					deadlocks = 0;
					RandomForest rf = new RandomForest();
					rf.setNumTrees(NUMBER_OF_TREES);
					Evaluation eval = new Evaluation(allInstances);
					
					rf.buildClassifier(trainingInstances);
					Classifier[] RFTrees = rf.getClassifiers();
					Vote committee = new Vote();
					Classifier[] committeeTrees = Functions.decideBestTrees(RFTrees, trainingInstances, COMMITTEE_SIZE, Functions.F1SCORE);
					committee.setClassifiers(committeeTrees);
					
					eval.evaluateModel(rf, testInstances);
					int tpRF = (int)eval.numTruePositives(POSITIVE);
					int fpRF = (int)eval.numFalsePositives(POSITIVE);
					int fnRF = (int)eval.numFalseNegatives(POSITIVE);
					double precRF = eval.precision(POSITIVE);
					double recRF = eval.recall(POSITIVE);
					double f1scoreRF = eval.fMeasure(POSITIVE);
					
					eval = new Evaluation(allInstances);
					eval.evaluateModel(committee, testInstances);
					int tpCM = (int)eval.numTruePositives(POSITIVE);
					int fpCM = (int)eval.numFalsePositives(POSITIVE);
					int tnCM = (int)eval.numTrueNegatives(POSITIVE);
					int fnCM = (int)eval.numFalseNegatives(POSITIVE);
					double precCM = eval.precision(POSITIVE);
					double recCM = eval.recall(POSITIVE);
					double f1scoreCM = eval.fMeasure(POSITIVE);
					double f2scoreCM = Functions.fScore(tpCM, tnCM, fpCM, fnCM, 2);
					//System.out.printf("^--------------- FOLD NUMBER %d ---------------^\n",i+1);
					
					
					List<Integer> deadlockInstances = new ArrayList<Integer>();
					for (int instID = 0; instID < testInstances.numInstances(); instID++){
						Instance inst = testInstances.instance(instID);
						//int real = (int)inst.classValue();
						//int pred = (int)rf.classifyInstance(inst);
						double[] probs = committee.distributionForInstance(inst);
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
					pw.printf("%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d",
							trainingInstances.numInstances(),tpCM,fpCM,fnCM,precCM,recCM,f1scoreCM,f2scoreCM,deadlocks);
					pw.printf("\t|\tRF\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\n",tpRF,fpRF,fnRF,precRF,recRF,f1scoreRF);
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
