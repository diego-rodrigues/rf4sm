package experiments.retraining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
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

public class UserCommitteeDriftingTraining {

	static int NUMBER_OF_TREES_GENERATED = 100;
	static double MIN_AGREEMENT = .75;
	static final String[] domainList = {"betting","business_partner","magazine_subscription","online_book","purchase_order"};
	static String EXPERIMENTS = "";
	static double TRAINING_PERCENTAGE = 1;
	static double USER_INTERVENTION_THRESHOLD = 0.75;
	static int MAX_USER_LABELS_PER_ROUND = 20;
	static double USER_COMMITTEE_MINIMUM_THRESHOLD = 0.75;
	
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
		for (randSeed = 0; randSeed < 1; randSeed++){
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
			String logFileName = EXPERIMENTS + "logs/UserCommitteeDriftingTraining/"+ domain + "/" + MIN_AGREEMENT + "-" + domain + "-" + Integer.toString(randSeed) + ".log.txt";
			try {
				pw = new PrintWriter(new File(logFileName));
				pw.printf("\t\t\t\tRF\t\t\t\n");
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
			
//			Functions.createRandomTrainingAndTest(trainingInstances, testInstances, allInstances, trainingPerc, r);			//Test#1
//			Functions.createCustomTrainingAndTest(trainingInstances, testInstances, allInstances, TRAINING_PERCENTAGE, 10, r);			//Test#2
			Functions.createFixedPNTrainingAndTest(trainingInstances, testInstances, allInstances, 5, 35, r);
			
			int maxPositives = Functions.countPositivesInSet(testInstances);
			try {
				Evaluation eval = new Evaluation(testInstances);
				
				
				for (int round = 0; (round < 10) && (testInstances.numInstances() > 0) && (trainingInstances.numInstances() > 0); round++){
					System.out.printf("%d ",round);
					int positivesInTraining = Functions.countPositivesInSet(trainingInstances);
					pw.printf("%d\t%d\t%d\t%d\t",randSeed,trainingInstances.numInstances(),positivesInTraining,
							testInstances.numInstances());
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
					double[] accuracyScores = new double[RFTrees.length];
					Vote rf = new Vote();
					rf.setClassifiers(RFTrees);
					Functions.castingVotes(rf, testInstances, predictedDecisionHistory, agreementHistory);		//all trees vote
					
					List<Integer> preliminarDecisions = Functions.getFinalDecisions(agreementHistory, MIN_AGREEMENT);
					List<Integer> finalDecisions = new ArrayList<Integer>();
					List<Integer> listOfUserLabelsInstancesIDs;
					listOfUserLabelsInstancesIDs =  															
							Functions.userLabelSplitDecisions(testInstances, predictedDecisionHistory, agreementHistory, USER_INTERVENTION_THRESHOLD, MAX_USER_LABELS_PER_ROUND);
					System.out.printf("User labeled %d instances.\n",listOfUserLabelsInstancesIDs.size());
					if (listOfUserLabelsInstancesIDs.size() > 0){
						Instances userLabeledInstances = new Instances(trainingInstances);
						userLabeledInstances.delete();
						Functions.copyInstancesWithPredictedLabel(listOfUserLabelsInstancesIDs, testInstances, userLabeledInstances, predictedDecisionHistory);
						double[] f1scores = new double[RFTrees.length];
						double[] f2scores = new double[RFTrees.length];
						double[] f0_5scores = new double[RFTrees.length];
						Functions.calcFScores(RFTrees, userLabeledInstances, f0_5scores, 2);
//						Functions.calcStatisticFunctions(RFTrees, usedLabeledInstances, f1scores, f2scores, accuracyScores); //only accuracy scores used in this test
//						for (int i=0; i<accuracyScores.length; i=i+4){
//							System.out.printf("%3d - %.3f\t%3d - %.3f\t%3d - %.3f\t%3d - %.3f\n",i,accuracyScores[i],i+1,accuracyScores[i+1],i+2,accuracyScores[i+2],i+3,accuracyScores[i+3]);
//						}
//						System.out.printf("----F1--------------------\n");
//						for (int i=0; i<accuracyScores.length; i=i+4){
//							System.out.printf("%3d - %.3f\t%3d - %.3f\t%3d - %.3f\t%3d - %.3f\n",i,f1scores[i],i+1,f1scores[i+1],i+2,f1scores[i+2],i+3,f1scores[i+3]);
//						}
//						System.out.printf("----F2--------------------\n");
//						for (int i=0; i<accuracyScores.length; i=i+4){
//							System.out.printf("%3d - %.3f\t%3d - %.3f\t%3d - %.3f\t%3d - %.3f\n",i,f2scores[i],i+1,f2scores[i+1],i+2,f2scores[i+2],i+3,f2scores[i+3]);
//						}
						System.out.printf("----F0.5--------------------\n");
						for (int i=0; i<f0_5scores.length; i=i+4){
							System.out.printf("%3d - %.3f\t%3d - %.3f\t%3d - %.3f\t%3d - %.3f\n",i,f0_5scores[i],i+1,f0_5scores[i+1],i+2,f0_5scores[i+2],i+3,f0_5scores[i+3]);
						}
//						committeeTrees = Functions.copyTreesAboveThreshold(RFTrees, f2scores, USER_COMMITTEE_MINIMUM_THRESHOLD);
						committeeTrees = Functions.copyTreesAboveThreshold(RFTrees, f0_5scores, USER_COMMITTEE_MINIMUM_THRESHOLD);
						System.out.printf("%d trees elected in committee\n-------------------\n",committeeTrees.length);
						
						rf.setClassifiers(committeeTrees);
						
						Functions.castingVotes(rf, testInstances, predictedDecisionHistory, agreementHistory);		//committee trees vote
						
//						finalDecisions = Functions.getFinalDecisions(agreementHistory, 1);
						finalDecisions = Functions.getFinalDecisionsByClass(agreementHistory, 1, predictedDecisionHistory, true);	//getting only positive predictions
						Functions.retainDecisions(preliminarDecisions, finalDecisions, agreementHistory);
						
						System.out.printf("alltrees decisions: %d\n",preliminarDecisions.size());
						System.out.printf("usercommittee decisions: %d\n",finalDecisions.size());
						System.out.printf("diff: %d\n",preliminarDecisions.size()-finalDecisions.size());
						finalDecisions.retainAll(preliminarDecisions);
						System.out.printf("final decisions: %d\n",finalDecisions.size());
					}else{
						finalDecisions = Functions.getFinalDecisions(agreementHistory, 1);
					}
					
					Functions.evaluateVotes(eval, testInstances, 1, predictedDecisionHistory, agreementHistory);
//					excecoes: 
//						*0 split decisions: 
//							*aumenta o desacordo minimo
//							*não pede auxilio ao usuario
//						*0 arvores concordando com o usuario
//							*diminui o acordo com o usuario
//							*cancela a iteracao, adiciona os rotulos do usuario no treino
					
					
					
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
					if (Double.isNaN(f1ScoreP)) f1ScoreP = 0;
					System.out.printf("%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t",precP,recP,f1ScoreP,TP,TN,FP,FN);
					
					pw.printf("%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t",precP,recP,f1ScoreP,TP,TN,FP,FN);

					pw.printf("%d\t", testInstances.numInstances());
					pw.printf("\n");
					break;
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			pw.printf("----------------------------------------\nUser Committee Drifting Training\n");
			pw.printf("Random seed: %d\n",randSeed);
			pw.printf("Number of candidate trees in the forest: %d\n" +
					"Minimum agreement required to label instance: %.3f\n" +
					"Maximum labels per round: %d\n",
					NUMBER_OF_TREES_GENERATED,MIN_AGREEMENT,MAX_USER_LABELS_PER_ROUND);
			pw.printf("Experiment finalized at %s\n",c.getTime().toString());
			long duration = System.currentTimeMillis() - startTime;
			pw.printf("Duration (s): %.2f\n", duration/1000.0);
			pw.close();
			System.out.printf("Done.\n");
		}
	}

}
