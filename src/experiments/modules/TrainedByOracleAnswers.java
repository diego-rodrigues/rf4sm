package experiments.modules;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class TrainedByOracleAnswers {
	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
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
//		BASELINE = SF;
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
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/modules/TrainedByOracleAnswers/" + baselineName + "-" + domainName +".txt";
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
			
			allInstances_COMA = instancesController.getPoolSet();		//pool set contains COMA extras attributes
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;

			List<Integer> trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> falseInstancesByOracle = new ArrayList<Integer>();
			List<Integer> inconsistentMatches = new ArrayList<Integer>();
			List<Integer> predictionsOracle = new ArrayList<Integer>();
			List<Integer> predictionsIDs = new ArrayList<Integer>();
			
			TRAINING_INSTANCES_NEGATIVE = allInstances.numInstances() / 10;
			if (TRAINING_INSTANCES_NEGATIVE < 2000) TRAINING_INSTANCES_NEGATIVE = 2000;
			
			TRAINING_INSTANCES_NEGATIVE = 2000;
			int jump = 4000;
			jump = 0;
			System.out.printf("Jump: %d\n",jump);
			
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, jump);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, truePredictionsBySFFileName, instancesController);
				Functions.requestInstancesFromSimFlood(falseInstancesByOracle, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, falsePredictionsBySFFileName, instancesController);
			}
			
//			int qtInstances = allInstances.numInstances();
//			List<Integer> falseInstancesTemp = new ArrayList<Integer>();
//			for (int instID = 0; instID < qtInstances; instID++){
//				if (!trueInstancesByOracle.contains(instID))
//					falseInstancesTemp.add(instID);
//			}
//			Random r = new Random(randSeed); 
//			Collections.shuffle(falseInstancesTemp, r);
//			falseInstancesByOracle.addAll(falseInstancesTemp.subList(0, TRAINING_INSTANCES_NEGATIVE));
			
			int numPred = trueInstancesByOracle.size();
			int quarto = numPred/4;
			
//			predictionsOracle.addAll(trueInstancesByOracle.subList(0, numPred-quarto));
			predictionsOracle.addAll(trueInstancesByOracle);
			
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Respostas #" + randSeed,pw);
			
			
			matchesController.restart();
			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, predictionsOracle);
			//MatchesFunctions.insertMatchesSequentialWithNoDerivingMatches(matchesController, instancesController,predictionsIDs);
			
			predictionsOracle = new ArrayList<Integer>();
			predictionsOracle.addAll(MatchesFunctions.getAcceptedMatches());
			
			Functions.measureStatistics(instancesController, predictionsOracle, baselineName + " Respostas+Rede#" + randSeed,pw);
			
			System.out.printf("Positives: %d\tNegatives: %d\n",predictionsOracle.size(),falseInstancesByOracle.size());
			int perc = 10;
			for (perc = 10; perc <= 100; perc = perc + 5){
				trueInstancesByOracle = predictionsOracle.subList(0, (int) (predictionsOracle.size()*(perc/100.0)));
				
				Instances trainingInstances = new Instances(allInstances);
				Instances testInstances = new Instances(allInstances);
				trainingInstances.delete();
				//Making up training set with predicted labels
				Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
				Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
				
				try {
					RandomForest randomForest = new RandomForest();
					randomForest.setSeed(randSeed);
					randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
	//				trainingInstances.deleteAttributeAt(0);
					randomForest.buildClassifier(trainingInstances);
	//				predictionsIDs = Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas do oraculo)",predictionsOracle,pw);
	//				System.out.printf("com prioridades: %s\n",predictionsIDs.toString());
					predictionsIDs = Functions.measureStatistics(testInstances, randomForest, "Aprendizado ("+perc+"%)",pw);
	//				System.out.printf("sem prioridades: %s\n",predictionsIDs.toString());
	//				System.exit(0);
					
				} catch (Exception e) {
				}
			}
			
			matchesController.restart();
			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, predictionsIDs);
			//MatchesFunctions.insertMatchesSequentialWithNoDerivingMatches(matchesController, instancesController,predictionsIDs);
			
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			//Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (conservador)",pw);
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (ordenado)\t",pw);
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
		}
	}
}
