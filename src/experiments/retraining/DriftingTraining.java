package experiments.retraining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class DriftingTraining {

	static int NUMBER_OF_TREES_COMMITTEE = 30;
	static int NUMBER_OF_TREES_GENERATED = 100;
	static double MIN_AGREEMENT = 1;
	static final String[] domainList = {"betting","business_partner","magazine_subscription","online_book","purchase_order"};
	static String EXPERIMENTS = "";
	static double TRAINING_PERCENTAGE = 1;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		Calendar c = new GregorianCalendar();
		String domain;
		Random r = null;
		int DOMAIN_ID = 4;		//book -- smaller domain
		
		if (args.length > 0){
			EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		int randSeed;
		for (randSeed = 0; randSeed < 30; randSeed++){
			domain = domainList[DOMAIN_ID];
			String arffFileName = EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + "-COMA-Matcher.arff";
			BufferedReader reader;
			Instances allInstances = null; 
			Instances trainingInstances = null;
			Instances testInstances = null;
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
			long startTime = System.currentTimeMillis(); 
			String logFileName = EXPERIMENTS + "logs/DriftingTraining/"+ domain + "/" + MIN_AGREEMENT + "-" + domain + "-" + NUMBER_OF_TREES_COMMITTEE + "-" + Integer.toString(randSeed) + ".log.txt";
			try {
				pw = new PrintWriter(new File(logFileName));
				pw.printf("\t\t\t\tRF(%d):Baseline\t\t\t\n",NUMBER_OF_TREES_COMMITTEE);
				pw.printf("Seed\t#Training\t#Tr-Positives\t#Test\t+Prec\t+Rec\t+F1Score\tTP\tTN\tFP\tFN\t#Unlabeled\n");
			} catch (FileNotFoundException e1) {
				System.err.printf("Error writing log (%s)\n",logFileName);
			}
			
			System.out.printf("Running %s (seed %d)\n",domain,randSeed);
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			r = new Random(randSeed);
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;
			String[] classes = new String[2]; 
			classes[Definitions.TRUE] = "TRUE";
			classes[Definitions.FALSE] = "FALSE";
//			System.out.printf("Positive aqui: %d\n",Definitions.TRUE);
			
			
//			Functions.createRandomTrainingAndTest(trainingInstances, testInstances, allInstances, trainingPerc, r);			//Test#1
//			Functions.createCustomTrainingAndTest(trainingInstances, testInstances, allInstances, TRAINING_PERCENTAGE, 10, r);			//Test#2
			Functions.createFixedPNTrainingAndTest(trainingInstances, testInstances, allInstances, 5, 35, r);
			
//			System.out.printf("Set size: %d\n",allInstances.numInstances());
//			System.out.printf("Positives in all set: %d (%s)\n",Functions.countPositivesInSet(allInstances),domain);
//			System.out.printf("Positives in training set: %d (%s)\n",Functions.countPositivesInSet(trainingInstances),domain);
//			System.out.printf("Positives in test set: %d (%s)\n",Functions.countPositivesInSet(testInstances),domain);
			int maxPositives = Functions.countPositivesInSet(testInstances);
//			System.out.printf("#RD\tRem+\tPr\tRc\tF1\tTP\tTN\tFP\tFN\tRemaining+\n");
			try {
//				Instances unlabeledInstances = new Instances(testInstances);
				Evaluation eval = new Evaluation(testInstances);
				
				
				for (int round = 0; (round < 10) && (testInstances.numInstances() > 0) && (trainingInstances.numInstances() > 0); round++){
//					System.out.printf("Positive aqui: %d\n",testInstances.classAttribute().indexOfValue("true"));
					System.out.printf("%d ",round);
					int positivesInTraining = Functions.countPositivesInSet(trainingInstances);
					pw.printf("%d\t%d\t%d\t%d\t",randSeed,trainingInstances.numInstances(),positivesInTraining,
							testInstances.numInstances());
//					System.out.printf("Train:%d\tTest:%d\t",trainingInstances.numInstances(),testInstances.numInstances());
//					System.out.printf("%d\t",Functions.countPositivesInSet(testInstances));
					//Creating Forest of Candidate Trees
					RandomForest candidateTrees = new RandomForest();
					candidateTrees.setNumTrees(NUMBER_OF_TREES_GENERATED);
					candidateTrees.buildClassifier(trainingInstances);
					
					Classifier[] RFTrees = candidateTrees.getClassifiers();
					
					Classifier[] committeeTrees;
					double precP,recP,f1ScoreP;
					int TP,TN,FP,FN;
					
					int[] predictedDecisionHistory = new int[testInstances.numInstances()];
					double[] agreementHistory = new double[testInstances.numInstances()];
					double[] fScores = new double[RFTrees.length];
					Vote rf = new Vote();
					
					Functions.calcFScores(RFTrees, trainingInstances, fScores, 0.5); 		//F0.5 to prioritize recall
					committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, fScores);
					rf.setClassifiers(committeeTrees);
					
					Functions.castingVotes(rf, testInstances, predictedDecisionHistory, agreementHistory);
					Functions.evaluateVotes(eval, testInstances, MIN_AGREEMENT, predictedDecisionHistory, agreementHistory);
					
					List<Integer> finalDecisions = Functions.getFinalDecisions(agreementHistory, MIN_AGREEMENT);
					
					trainingInstances.delete();
					Functions.copyInstancesWithPredictedLabel(finalDecisions, testInstances, trainingInstances, predictedDecisionHistory);
					Functions.removeInstancesFromSet(finalDecisions, testInstances);
					
					precP = eval.precision(Definitions.TRUE);
					recP = eval.recall(Definitions.TRUE);
					f1ScoreP = eval.fMeasure(Definitions.TRUE);
					TP = (int)eval.numTruePositives(Definitions.TRUE);
					TN = (int)eval.numTrueNegatives(Definitions.TRUE);
					FP = (int)eval.numFalsePositives(Definitions.TRUE);
					FN = (int)eval.numFalseNegatives(Definitions.TRUE);
					recP = TP/(double)(maxPositives);
					f1ScoreP = (2*precP*recP)/(precP+recP);
					
					pw.printf("%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t",precP,recP,f1ScoreP,TP,TN,FP,FN);
//					System.out.printf("%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t",precP,recP,f1ScoreP,TP,TN,FP,FN);
//					System.out.printf("%d\n",Functions.countPositivesInSet(testInstances));

					pw.printf("%d\t", testInstances.numInstances());
					pw.printf("\n");
					
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			pw.printf("\n----------------------------------------\nSmall Training Retraining Committee (F0.5)\n");
			pw.printf("Random seed: %d\n",randSeed);
			pw.printf("Number of Random Forest trees (baseline): %d\n" +
					"Number of candidate trees in the forest: %d\n" +
					"Minimum agreement required to label instance: %.3f\n",
					NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_GENERATED,MIN_AGREEMENT);
			pw.printf("Experiment finalized at %s\n",c.getTime().toString());
			long duration = System.currentTimeMillis() - startTime;
			pw.printf("Duration (s): %.2f\n", duration/1000.0);
			pw.close();
			System.out.printf("Done.\n");
		}
	}

}