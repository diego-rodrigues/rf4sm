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
 * Experimento que avalia o desempenho de uma RandomForest contra o de um comite de mesmo tamanho gerado através do F1, 
 * F2, Acuracia. 
 * @author diego
 *
 */
public class SupervisedForestVsCommittee2 {

	static int NUMBER_OF_TREES_COMMITTEE = 10;
	static int NUMBER_OF_TREES_GENERATED = 100;
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
		long startTime = System.currentTimeMillis(); 
		String logFileName = EXPERIMENTS + "logs/SupervisedForestVsCommittee2/"+ domain + "/" + domain + "-" + NUMBER_OF_TREES_COMMITTEE + ".log.txt";
		try {
			pw = new PrintWriter(new File(logFileName));
			pw.printf("\t\t\tRF(%d):Baseline\t \t \t" +
					"RF(%d):All Trees\t \t \t" +
					"Committee-%d-F1\t \t \t" +
					"Committee-%d-F2\t \t \t" +
					"Committee-%d-Accuracy\n",
					NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_GENERATED,NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_COMMITTEE);
			pw.printf("Seed\t#Training\t#Test\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\tPrec\tRec\tF1Score\n");
		} catch (FileNotFoundException e1) {
			System.err.printf("Error writing log (%s)\n",logFileName);
		}
		
		double[] counters = new double[15];
		int[] highScore = new int[5];
		int randSeed;
		for (randSeed = 0; randSeed < 30; randSeed++){
			System.out.printf("Running %s (seed %d)\n",domain,randSeed);
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			r = new Random(randSeed);
			allInstances.randomize(r); 
			int POSITIVE = allInstances.classAttribute().indexOfValue("true");
			int NEGATIVE = 1 - POSITIVE;
			String[] classes = new String[2]; 
			classes[POSITIVE] = "TRUE";
			classes[NEGATIVE] = "FALSE";
			
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
			try {
				pw.printf("%d\t%d\t%d\t",randSeed,trainingInstances.numInstances(),testInstances.numInstances());
				//Creating Forest of Candidate Trees
				RandomForest candidateTrees = new RandomForest();
				candidateTrees.setNumTrees(NUMBER_OF_TREES_GENERATED);
				candidateTrees.buildClassifier(trainingInstances);
				double[] finalScores = new double[5];
				double maxScore = 0;
				
				//Calculating F1, F2, and Accuracy scores of Candidate Trees
				double[] f1Scores = new double[NUMBER_OF_TREES_GENERATED];
				double[] f2Scores = new double[NUMBER_OF_TREES_GENERATED];
				double[] accuracyScores = new double[NUMBER_OF_TREES_GENERATED];
				Classifier[] RFTrees = candidateTrees.getClassifiers();
				Functions.calcStatisticFunctions(RFTrees, trainingInstances, f1Scores, f2Scores, accuracyScores);
				
				//Random Forest(K)-Baseline: evaluation
				Vote rf = new Vote();
				Classifier[] committeeTrees = Functions.copyKTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE);
				rf.setClassifiers(committeeTrees);
				Evaluation eval = new Evaluation(testInstances);
				eval.evaluateModel(rf, testInstances);
				double prec = eval.precision(POSITIVE);
				double rec = eval.recall(POSITIVE);
				double f1Score = eval.fMeasure(POSITIVE);
				pw.printf("%.3f\t%.3f\t%.3f\t", prec,rec,f1Score);
				finalScores[0] = f1Score;
				maxScore = f1Score;
				counters[0] += prec;
				counters[1] += rec;
				counters[2] += f1Score;
				
				//Random Forest All Candidates: evaluation
				eval = new Evaluation(testInstances);
				eval.evaluateModel(candidateTrees, testInstances);
				prec = eval.precision(POSITIVE);
				rec = eval.recall(POSITIVE);
				f1Score = eval.fMeasure(POSITIVE);
				pw.printf("%.3f\t%.3f\t%.3f\t", prec,rec,f1Score);
				finalScores[1] = f1Score;
				if (f1Score > maxScore) maxScore = f1Score;
				counters[3] += prec;
				counters[4] += rec;
				counters[5] += f1Score;
				
				//F1-Score Committee: evaluation
				Vote committee = new Vote();
				committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, f1Scores);
				committee.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				eval.evaluateModel(committee, testInstances);
				prec = eval.precision(POSITIVE);
				rec = eval.recall(POSITIVE);
				f1Score = eval.fMeasure(POSITIVE);
				pw.printf("%.3f\t%.3f\t%.3f\t", prec,rec,f1Score);
				finalScores[2] = f1Score;
				if (f1Score > maxScore) maxScore = f1Score;
				counters[6] += prec;
				counters[7] += rec;
				counters[8] += f1Score;
				
				//F2-Score Committee: evaluation
				committee = new Vote();
				committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, f2Scores);
				committee.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				eval.evaluateModel(committee, testInstances);
				prec = eval.precision(POSITIVE);
				rec = eval.recall(POSITIVE);
				f1Score = eval.fMeasure(POSITIVE);
				pw.printf("%.3f\t%.3f\t%.3f\t", prec,rec,f1Score);
				finalScores[3] = f1Score;
				if (f1Score > maxScore) maxScore = f1Score;
				counters[9] += prec;
				counters[10] += rec;
				counters[11] += f1Score;
				
				//Accuracy-Score Committee: evaluation
				committee = new Vote();
				committeeTrees = Functions.decideBestTrees(RFTrees, NUMBER_OF_TREES_COMMITTEE, accuracyScores);
				committee.setClassifiers(committeeTrees);
				eval = new Evaluation(testInstances);
				eval.evaluateModel(committee, testInstances);
				prec = eval.precision(POSITIVE);
				rec = eval.recall(POSITIVE);
				f1Score = eval.fMeasure(POSITIVE);
				pw.printf("%.3f\t%.3f\t%.3f\t", prec,rec,f1Score);
				finalScores[4] = f1Score;
				if (f1Score > maxScore) maxScore = f1Score;
				counters[12] += prec;
				counters[13] += rec;
				counters[14] += f1Score;
				
				for (int i = 0; i < 5; i++){
					if (finalScores[i] == maxScore){
						pw.printf("X\t");
						highScore[i]++;
					}
					else pw.printf("_\t");
				}
				pw.printf("\n");
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		pw.printf("avg\t\t\t");
		for (int i = 0; i < counters.length; i++){
			counters[i] = counters[i] / (double)(randSeed);
			pw.printf("%.3f\t",counters[i]);
		}
		for (int i = 0; i < highScore.length; i++)
			pw.printf("%d\t", highScore[i]);
		pw.printf("\n");
		pw.printf("\n----------------------------------------\nSupervised Forest VS Committee - 2\n");
		pw.printf("Number of Random Forest trees (baseline): %d\n" +
				"Number of candidate trees in the forest: %d\n",
				NUMBER_OF_TREES_COMMITTEE,NUMBER_OF_TREES_GENERATED);
		pw.printf("Experiment finalized at %s\n",c.getTime().toString());
		long duration = System.currentTimeMillis() - startTime;
		pw.printf("Duration (s): %.2f\n", duration/1000.0);
		pw.close();
		System.out.printf("Done.\n");
	}

}
