package experiments.sandbox;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class LearningOnFirstNegativeExamples {
	static int TRAINING_INSTANCES_POSITIVE = 800;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 50;				//default 50
	static final int COMA = 0;
	static final int SF = 1;
	static int BASELINE = COMA;
	static boolean SANDBOX = true; 
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		
		String domainName;
		String baselineName = "COMA";
		String logFileName;
		File logFile = null;
		PrintWriter pw = null;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
//		DOMAIN_ID = Definitions.BUSINESS;	
//		DOMAIN_ID = Definitions.MAGAZINE;	
//		DOMAIN_ID = Definitions.BOOK;
//		DOMAIN_ID = Definitions.ORDER;
		BASELINE = SF;
		//--------------------- command line options
		//command java -cp ../lib/wekagp-diego.jar:../bin experiments.network.user.ReconciliationAfterTraining 2 2 &
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n" +
					"> Domain (1-betting | 2-business | 3-magazine | 4-book | 5-order)\n" +
					"> Baseline (1-COMA | 2-Similarity Flooding)\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			BASELINE = Integer.valueOf(args[1]) - 1;
			SANDBOX = false;
		}
		//--------------------- 
		//--------------------- initialization
		if (BASELINE == SF){
			baselineName = "SF";
			TRAINING_INSTANCES_POSITIVE = 800;
		}
		
		for (DOMAIN_ID = 0; DOMAIN_ID < 5; DOMAIN_ID++){
			long startTime = System.currentTimeMillis();
			int randSeed;
			randSeed = 0;
			domainName = Definitions.DOMAIN_LIST[DOMAIN_ID];
			
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			Instances allInstances = null;
			Instances allInstances_COMA = null;
			//--------------------- 
			//--------------------- Log settings
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/modules/JustOracle/" + baselineName + "-" + domainName +".txt";
			try {
				if (!SANDBOX){
					logFile = new File(logFileName);
					pw = new PrintWriter(logFile);
				}
			} catch (FileNotFoundException e1) {
				System.err.printf("Error writing logfile [%s]\n",logFileName);
			}
			if (!SANDBOX){
				pw.printf("Domain: %s\n",domainName);	
				pw.printf("Experiment#Seed\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
			}else{
				System.out.printf("Domain: %s\n",domainName);	
				System.out.printf("Experiment#Seed\t\t\t\t|\tPREC\tREC\tF1SCORE\tTP\tTN\tFP\tFN\n");
			}
			//---------------------
			if (!SANDBOX) System.out.printf("|%s-%s-%d|\n",baselineName,domainName,randSeed);
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domainName + "-COMA_AvgMatcher.arff";
			String truePredictionsBySFFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-true-predictions_" + domainName + ".txt";
			String falsePredictionsBySFFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-false-predictions_" + domainName + ".txt";
			
			instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
			matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
			
			allInstances_COMA = instancesController.getPoolSet();				//pool set contains COMA extras attributes
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;

			List<Integer> trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> falseInstancesByOracle = new ArrayList<Integer>();
			
			TRAINING_INSTANCES_NEGATIVE = allInstances.numInstances() / 5;
			if (TRAINING_INSTANCES_NEGATIVE < 2000) TRAINING_INSTANCES_NEGATIVE = 2000;
			int jump = 4000;
			System.out.printf("Jump: %d\n",jump);
			
			List<Integer> firstNotPositiveIDs = new ArrayList<Integer>();
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, jump);
				Functions.requestInstancesFromCOMA(allInstances_COMA, firstNotPositiveIDs, Definitions.FALSE, jump, Functions.DESCENDING);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, truePredictionsBySFFileName, instancesController);
				Functions.requestInstancesFromSimFlood(falseInstancesByOracle, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, falsePredictionsBySFFileName, instancesController, jump);
				Functions.requestInstancesFromSimFlood(firstNotPositiveIDs, jump, Functions.DESCENDING, falsePredictionsBySFFileName, instancesController, 0);
			}
			Instances firstNotPositiveSet = Functions.createSetOfInstances(allInstances, firstNotPositiveIDs);
			
			Instances train = new Instances(allInstances);
			train.delete();
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, train, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, train, Definitions.FALSE);
			
			RandomForest randomForest = new RandomForest();
			try {
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(train);
				Evaluation ev = new Evaluation(firstNotPositiveSet);
				ev.evaluateModel(randomForest, firstNotPositiveSet);
				double precision = ev.precision(Definitions.TRUE);
				double recall = ev.recall(Definitions.TRUE);
				double f1_score = ev.fMeasure(Definitions.TRUE);
				System.out.printf("\n============================================\n");
				System.out.printf("Number of positives: %d (of 4000).\n",Functions.countPositivesInSet(firstNotPositiveSet));
				System.out.printf("Precision: %.3f\n",precision);
				System.out.printf("Recall: %.3f\n",recall);
				System.out.printf("F1-Score: %.3f\n",f1_score);
			} catch (Exception e) {
			}
			
			//Cross Validation test
			/*RandomForest randomForest = new RandomForest();
			firstNotPositiveSet.stratify(10);
			double avgP = 0.0;
			double avgR = 0.0;
			double avgF = 0.0;
			for (int foldNo = 0; foldNo < 10; foldNo++){
				Instances test = firstNotPositiveSet.trainCV(10, foldNo);
				Instances train = firstNotPositiveSet.testCV(10, foldNo);
				try {
					randomForest.setSeed(randSeed);
					randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
					randomForest.buildClassifier(train);
//					System.out.printf("Size: Training: %d\tTest: %d\n",train.numInstances(),test.numInstances());
					Evaluation ev = new Evaluation(test);
					ev.evaluateModel(randomForest, test);
//					System.out.printf("Results (fold %d):\n%s\n",foldNo,ev.toSummaryString());
					double precision = ev.precision(Definitions.TRUE);
					double recall = ev.recall(Definitions.TRUE);
					double f1_score = ev.fMeasure(Definitions.TRUE);
					avgP += precision;
					avgR += recall;
					avgF += f1_score;
//					System.out.printf("Precision: %.3f\n",precision);
//					System.out.printf("Recall: %.3f\n",recall);
//					System.out.printf("F1-Score: %.3f\n",f1_score);
				} catch (Exception e) {
				}
			}
			avgP = avgP / 10.0;
			avgR = avgR / 10.0;
			avgF = avgF / 10.0;*/
//			System.out.printf("\n============================================\n");
//			System.out.printf("Number of positives: %d (of 4000).\n",Functions.countPositivesInSet(firstNotPositiveSet));
//			System.out.printf("Precision: %.3f\n",avgP);
//			System.out.printf("Recall: %.3f\n",avgR);
//			System.out.printf("F1-Score: %.3f\n",avgF);
			
			Calendar c = new GregorianCalendar();
			long duration = System.currentTimeMillis() - startTime;
			if (!SANDBOX){
//				pw.printf("User(+): %d\n" +
//					"User(-): %d\n" +
//					"Rede(+): %d\n" +
//					"Rede(-): %d\n",MatchesFunctions.getUserConfirmedMatches().size(),
//					MatchesFunctions.getUserRejectedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size(),
//					MatchesFunctions.getNetworkRejectedMatches().size());
				
				pw.printf("\nExperiment run at: %s\n",c.getTime());
				pw.printf("Duration (s): %.2f\n", duration/1000.0);
				pw.close();
			}else{
//				System.out.printf("User(+): %d\n" +
//						"User(-): %d\n" +
//						"Rede(+): %d\n" +
//						"Rede(-): %d\n",MatchesFunctions.getUserConfirmedMatches().size(),
//						MatchesFunctions.getUserRejectedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size(),
//						MatchesFunctions.getNetworkRejectedMatches().size());
				System.out.printf("\nExperiment run at: %s\n",c.getTime());
				System.out.printf("Duration (s): %.2f\n", duration/1000.0);
				System.out.printf("\n");
			}
//			System.exit(0);
		}
	}
}
