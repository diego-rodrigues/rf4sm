package experiments.supervised;
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
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.classifiers.functions.LibSVM;
import experiments.Definitions;
import experiments.functions.Functions;

public class SupervisedTests {
	static final String[] domainList = {"betting","business_partner","magazine_subscription","online_book","purchase_order"};
	static String EXPERIMENTS = "";
	static double TRAINING_PERCENTAGE = 40;
	
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
//			String arffFileName = EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + "-COMA-Matcher.arff";
			String arffFileName = EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + ".arff";
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
//			String logFileName = EXPERIMENTS + "logs/SupervisedTests/training" + (int)(TRAINING_PERCENTAGE) + "/" + domainList[DOMAIN_ID] + "-" + Integer.toString(randSeed) + ".log.txt";
			String logFileName = EXPERIMENTS + "logs/sandbox.log.txt";
			try {
				pw = new PrintWriter(new File(logFileName));
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
			
			Functions.createRandomTrainingAndTest(trainingInstances, testInstances, allInstances, TRAINING_PERCENTAGE, r);			//Test#1
//			Functions.createCustomTrainingAndTest(trainingInstances, testInstances, allInstances, TRAINING_PERCENTAGE, 10, r);			//Test#2
//			Functions.createFixedPNTrainingAndTest(trainingInstances, testInstances, allInstances, 15, 35, r);
			
			String[] classifiersNames = {"J48","AdaBoost","Logistic","RandomTree","RndForest"};
			Classifier[] classifiers = new Classifier[6];
			classifiers[0] = new J48();
			classifiers[1] = new AdaBoostM1();
			classifiers[2] = new Logistic();
			classifiers[3] = new RandomTree();
			classifiers[4] = new RandomForest();
			
			Evaluation eval = null;
			pw.printf("%10s\t%7s\t%7s\t%7s\t%7s\t|\t%4s\t%4s\t%6s\t%6s\t" +
					"%10s\t%7s\t%7s\t%7s\t%7s\t|\t%4s\t%4s\t%6s\t%6s\n","Classifier","PRECIS","RECALL","F1-SCO",
					"ACCURA","TP","FP","TN","FN","Ev.AllInst","PRECIS","RECALL","F1-SCO","ACCURA","TP","FP","TN","FN");
//		System.out.printf("\n" +
//				"%10s\t%7s\t%7s\t%7s\t%7s\t|\t%4s\t%4s\t%6s\t%6s\n","Classifier","PRECIS","RECALL","F1-SCO","ACCURA","TP","FP","TN","FN");
			for (int i=0; i<5; i++){
				try {
					eval = new Evaluation(testInstances);
				} catch (Exception e1) {
					System.err.printf("Error generating evaluation module for classifier %s.\n",classifiersNames[i]);
				}
				try {
					classifiers[i].buildClassifier(trainingInstances);
				} catch (Exception e) {
					System.err.printf("Error while classifiyng by %s.\n",classifiersNames[i]);
				}
				try {
					eval.evaluateModel(classifiers[i], testInstances);
					pw.printf("%10s\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t|\t%4d\t%4d\t%6d\t%6d\t",classifiersNames[i],
							eval.precision(Definitions.TRUE),
							eval.recall(Definitions.TRUE),
							eval.fMeasure(Definitions.TRUE),
							eval.pctCorrect()/100.0,
							(int)eval.numTruePositives(Definitions.TRUE),
							(int)eval.numFalsePositives(Definitions.TRUE),
							(int)eval.numTrueNegatives(Definitions.TRUE),
							(int)eval.numFalseNegatives(Definitions.TRUE));
					eval = new Evaluation(allInstances);
					eval.evaluateModel(classifiers[i], allInstances);
					pw.printf("%10s\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t|\t%4d\t%4d\t%6d\t%6d\n",classifiersNames[i],
							eval.precision(Definitions.TRUE),
							eval.recall(Definitions.TRUE),
							eval.fMeasure(Definitions.TRUE),
							eval.pctCorrect()/100.0,
							(int)eval.numTruePositives(Definitions.TRUE),
							(int)eval.numFalsePositives(Definitions.TRUE),
							(int)eval.numTrueNegatives(Definitions.TRUE),
							(int)eval.numFalseNegatives(Definitions.TRUE));
	//				System.out.printf("%10s\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t|\t%4d\t%4d\t%6d\t%6d\n",classifiersNames[i],
	//						eval.precision(Definitions.TRUE),
	//						eval.recall(Definitions.TRUE),
	//						eval.fMeasure(Definitions.TRUE),
	//						eval.pctCorrect()/100.0,
	//						(int)eval.numTruePositives(Definitions.TRUE),
	//						(int)eval.numFalsePositives(Definitions.TRUE),
	//						(int)eval.numTrueNegatives(Definitions.TRUE),
	//						(int)eval.numFalseNegatives(Definitions.TRUE));
				} catch (Exception e) {
					System.err.printf("Error while evaluating %s.\n",classifiersNames[i]);
				}
			}
			pw.printf("\n----------------------------------------\nSupervisedTests\n");
			pw.printf("Random seed: %d\n",randSeed);
			pw.printf("Experiment finalized at %s\n",c.getTime().toString());
			long duration = System.currentTimeMillis() - startTime;
			pw.printf("Duration (s): %.2f\n", duration/1000.0);
			pw.close();
			System.out.printf("Finished %s\n",arffFileName);
		}
	}

}
