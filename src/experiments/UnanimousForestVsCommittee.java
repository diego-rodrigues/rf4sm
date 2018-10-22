package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.functions.Functions;

public class UnanimousForestVsCommittee {

	static int NUMBER_OF_TREES_COMMITTEE = 10;
	static int NUMBER_OF_TREES_GENERATED = 100;
	static double MIN_AGREEMENT = 1;
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
					"> Forest/Committee Size (10, 20, 30, 40, 50)\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			NUMBER_OF_TREES_COMMITTEE = Integer.valueOf(args[1]);
		}
		domain = domainList[DOMAIN_ID];
		String arffFileName = EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + "-COMA-Matcher.arff";
		double trainingPerc = 1;
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
		String logFileName = EXPERIMENTS + "logs/UnanimousForestVsCommittee/"+ domain + "/" + MIN_AGREEMENT + "- " + domain + "-" + NUMBER_OF_TREES_COMMITTEE + ".log.txt";
		try {
			pw = new PrintWriter(new File(logFileName));
			pw.printf("\t\t\t\tRF(%d):Baseline\t\t\t\t\t\t" +
					"Committee-%d-F1\t\t\t\t\t\t" +
					"Committee-%d-F2\t\t\t\t\t\n",
					NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_COMMITTEE);
			pw.printf("Seed\t#Training\t#Tr-Positives\t#Test\t+Prec\t+Rec\t+F1Score\t-Prec\t-Rec\t-F1Score\t" +
					"+Prec\t+Rec\t+F1Score\t-Prec\t-Rec\t-F1Score\t+Prec\t+Rec\t+F1Score\t-Prec\t-Rec\t-F1Score\t#+Discovered\n");
		} catch (FileNotFoundException e1) {
			System.err.printf("Error writing log (%s)\n",logFileName);
		}
		
		double[] counters = new double[19];
		int randSeed;
		for (randSeed = 0; randSeed < 30; randSeed++){
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
			Functions.createCustomTrainingAndTest(trainingInstances, testInstances, allInstances, trainingPerc, 10, r);			//Test#2
			try {
				int positivesInTraining = Functions.countPositivesInSet(trainingInstances);
				pw.printf("%d\t%d\t%d\t%d\t",randSeed,trainingInstances.numInstances(),positivesInTraining,
						testInstances.numInstances());
				//Creating Forest of Candidate Trees
				RandomForest candidateTrees = new RandomForest();
				candidateTrees.setNumTrees(NUMBER_OF_TREES_GENERATED);
				candidateTrees.buildClassifier(trainingInstances);
				
				//Calculating F1, F2, and Accuracy scores of Candidate Trees
				double[] f1Scores = new double[NUMBER_OF_TREES_GENERATED];
				double[] f2Scores = new double[NUMBER_OF_TREES_GENERATED];
				double[] accuracyScores = new double[NUMBER_OF_TREES_GENERATED];
				Classifier[] RFTrees = candidateTrees.getClassifiers();
				Functions.calcStatisticFunctions(RFTrees, trainingInstances, f1Scores, f2Scores, accuracyScores);
				
				Classifier[] committeeTrees;
				Evaluation eval;
				double precP,recP,f1ScoreP,precN,recN,f1ScoreN;
				int discP;
				
				//Random Forest(K)-Baseline: evaluation
				Vote rf = new Vote();
				committeeTrees = Functions.copyKTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE);
				rf.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				Functions.evaluateVotes(eval, rf, testInstances,MIN_AGREEMENT);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				precN = eval.precision(Definitions.FALSE);
				recN = eval.recall(Definitions.FALSE);
				f1ScoreN = eval.fMeasure(Definitions.FALSE);
				discP = (int)eval.numTruePositives(Definitions.TRUE);
				pw.printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t",precP,recP,f1ScoreP,precN,recN,f1ScoreN);
				counters[0] += precP;
				counters[1] += recP;
				counters[2] += f1ScoreP;
				counters[3] += precN;
				counters[4] += recN;
				counters[5] += f1ScoreN;
				counters[18] += discP;
				
				//F1-Score Committee: evaluation
				Vote committee = new Vote();
				committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, f1Scores);
				committee.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				Functions.evaluateVotes(eval, committee, testInstances,MIN_AGREEMENT);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				precN = eval.precision(Definitions.FALSE);
				recN = eval.recall(Definitions.FALSE);
				f1ScoreN = eval.fMeasure(Definitions.FALSE);
				pw.printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t",precP,recP,f1ScoreP,precN,recN,f1ScoreN);
				counters[6] += precP;
				counters[7] += recP;
				counters[8] += f1ScoreP;
				counters[9] += precN;
				counters[10] += recN;
				counters[11] += f1ScoreN;
				
				//F2-Score Committee: evaluation
				committee = new Vote();
				committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, f2Scores);
				committee.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				Functions.evaluateVotes(eval, committee, testInstances,MIN_AGREEMENT);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				precN = eval.precision(Definitions.FALSE);
				recN = eval.recall(Definitions.FALSE);
				f1ScoreN = eval.fMeasure(Definitions.FALSE);
				pw.printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t",precP,recP,f1ScoreP,precN,recN,f1ScoreN);
				counters[12] += precP;
				counters[13] += recP;
				counters[14] += f1ScoreP;
				counters[15] += precN;
				counters[16] += recN;
				counters[17] += f1ScoreN;

				pw.printf("%d\t", discP);
				pw.printf("\n");
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		pw.printf("avg\t\t\t\t");
		for (int i = 0; i < counters.length; i++){
			counters[i] = counters[i] / (double)(randSeed);
			pw.printf("%.3f\t",counters[i]);
		}
		pw.printf("\n");
		pw.printf("\n----------------------------------------\nUnanimous Forest VS Committee\n");
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
