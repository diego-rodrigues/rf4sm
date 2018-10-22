package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.GregorianCalendar;
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

/**
 * Experimento que avalia o desempenho de uma RandomForest(10) contra o comitê gerado através do F1. 
 * Decisões divididas da floresta são repassados ao oráculo e, depois de rotulados, incorporados ao conjunto de treino.
 * @author diego
 *
 */
public class ForestVsCommittee {

	static int COMMITTEE_SIZE = 10;
	static int NUMBER_OF_TREES_IN_FOREST = 10;
	static int NUMBER_OF_TREES_GENERATED = 100;
	static double MIN = 0.5;
	static double MAX = 0.5;
	static final String[] domainList = {"betting","business_partner","magazine_subscription","online_book","purchase_order"};
	static String EXPERIMENTS = "";
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		Calendar c = new GregorianCalendar();
		String domain;
		Random r = null;
		int DOMAIN_ID = 3;		//book -- smaller domain
		if (args.length > 0){
			EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n" +
					"> Forest/Committee Size (10, 20, 50)\n" +
					"> Upper and lower bound of disagreement (default: 0.5 0.5)\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			COMMITTEE_SIZE = Integer.valueOf(args[1]);
			NUMBER_OF_TREES_IN_FOREST = COMMITTEE_SIZE;
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
			long startTime = System.currentTimeMillis(); 
			String logFileName = EXPERIMENTS + "logs/ForestVsCommittee" + COMMITTEE_SIZE + "/"+ domain + "/" + domain + "-" + randSeed + ".log.txt";
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
			pw.printf("RandomForest(%d)\tCommittee-%d-F1\tCommittee-%d-F2\tAllForest(%d)\n",
					NUMBER_OF_TREES_IN_FOREST,NUMBER_OF_TREES_IN_FOREST,NUMBER_OF_TREES_IN_FOREST,NUMBER_OF_TREES_GENERATED);
			pw.printf("#Training\tTP\tFP\tFN\tPrec\tRec\tF1Score\t#Deadlocks\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\n");
			
			int rounds = 0;
			int sumDeadlocks = 0;
			try {
				int deadlocks = 0;
				do{
					deadlocks = 0;
					int trainingSize = trainingInstances.numInstances();
					//Creating Random Forest
					RandomForest rf = new RandomForest();
					rf.setNumTrees(NUMBER_OF_TREES_IN_FOREST);
					rf.buildClassifier(trainingInstances);
					
					//Creating Forest of Candidate Trees
					RandomForest candidateTrees = new RandomForest();
					candidateTrees.setNumTrees(NUMBER_OF_TREES_GENERATED);
					candidateTrees.buildClassifier(trainingInstances);
					
					//Calculating F1 and F2 scores of Candidate Trees
					double[] f1Scores = new double[NUMBER_OF_TREES_GENERATED];
					double[] f2Scores = new double[NUMBER_OF_TREES_GENERATED];
					Classifier[] RFTrees = candidateTrees.getClassifiers();
					Functions.calcFScores(RFTrees, trainingInstances, f1Scores, f2Scores);
					
					//Selecting Committees of Trees based on F1-score and F2-score
					Vote committeeF1 = new Vote();
					Vote committeeF2 = new Vote();
					Classifier[] committeeTrees = Functions.decideBestTrees(RFTrees, COMMITTEE_SIZE, f1Scores);
					committeeF1.setClassifiers(committeeTrees);
					committeeTrees = Functions.decideBestTrees(RFTrees, COMMITTEE_SIZE, f2Scores);
					committeeF2.setClassifiers(committeeTrees);
					
					//Evaluating Random Forest and Recording Split Decisions (deadlock)
					List<Integer> deadlockInstances = new ArrayList<Integer>();
					Evaluation eval = new Evaluation(testInstances);
					for (int instID = 0; instID < testInstances.numInstances(); instID++){
						Instance inst = testInstances.instance(instID);
						double[] probs = rf.distributionForInstance(inst);
						if ((probs[0] >= MIN) && (probs[0] <= MAX)){
							deadlocks++;
							deadlockInstances.add(instID);
							//System.out.printf("instance %d no decision\n",instID);
						}
						eval.evaluateModelOnce(probs, inst);
					}
					double RFPrec = eval.precision(POSITIVE);
					double RFRec = eval.recall(POSITIVE);
					double RFF1 = eval.fMeasure(POSITIVE);
					int TP = (int)eval.numTruePositives(POSITIVE);
					int FP = (int)eval.numFalsePositives(POSITIVE);
					int FN = (int)eval.numFalseNegatives(POSITIVE);
					
					//Evaluating Committee based on F1Scores
					eval = new Evaluation(testInstances);
					eval.evaluateModel(committeeF1, testInstances);
					double CMF1Prec = eval.precision(POSITIVE);
					double CMF1Rec = eval.recall(POSITIVE);
					double CMF1F1 = eval.fMeasure(POSITIVE);
					
					//Evaluating Committee based on F2Scores
					eval = new Evaluation(testInstances);
					eval.evaluateModel(committeeF2, testInstances);
					double CMF2Prec = eval.precision(POSITIVE);
					double CMF2Rec = eval.recall(POSITIVE);
					double CMF2F1 = eval.fMeasure(POSITIVE);
					
					//Evaluating Candidate Trees as a Forest
					eval = new Evaluation(testInstances);
					eval.evaluateModel(candidateTrees, testInstances);
					double CTPrec = eval.precision(POSITIVE);
					double CTRec = eval.recall(POSITIVE);
					double CTF1 = eval.fMeasure(POSITIVE);
					
					//Labeling process. Adding instances to training set of instances
					Collections.sort(deadlockInstances);
					for (int i = deadlocks-1; i >= 0; i--){
						int instanceID = deadlockInstances.get(i);
						Instance inst = testInstances.instance(instanceID);
						testInstances.delete(instanceID);
						trainingInstances.add(inst);
						//System.out.printf("Instance %d added to training\n",instanceID);
					}
					
					pw.printf("%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
							trainingSize,TP,FP,FN,RFPrec,RFRec,RFF1,deadlocks,CMF1Prec,CMF1Rec,CMF1F1,CMF2Prec,CMF2Rec,CMF2F1,
							CTPrec,CTRec,CTF1);
					
					rounds++;
					sumDeadlocks += deadlocks;
					System.out.printf("%d ",rounds);
					//System.out.printf("----------------------------------------------\n\n");
				}while ((deadlocks > 0) && (testInstances.numInstances() > 0));
			} catch (Exception e) {
				e.printStackTrace();
			}
			pw.printf("\n----------------------------------------\nForest VS Committee\n");
			pw.printf("Number of Random Forest/Committee trees: %d\n" +
					"Number of candidate trees in the forest: %d\n" +
					"Number of iterations with oracle: %d\n" +
					"Number of oracle labels: %d\n" +
					"Total of training: %d\n",
					NUMBER_OF_TREES_IN_FOREST,NUMBER_OF_TREES_GENERATED,rounds,sumDeadlocks,trainingInstances.numInstances());
			pw.printf("Experiment finalized at %s\n",c.getTime().toString());
			long duration = System.currentTimeMillis() - startTime;
			pw.printf("Duration (s): %.2f\n", duration/1000.0);
			pw.close();
			System.out.printf("Done.\n");
		}
	}

}
