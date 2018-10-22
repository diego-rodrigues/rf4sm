package experiments.supervised;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class SupervisedByOracleSF {

	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 50;				//default 50
	static final int COMA = 0;
	static final int SF = 1;
	static int BASELINE = COMA;
	static boolean SANDBOX = true; 
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		
		String domain;
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
		//options when running from command line
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
		
		//initialization
		if (BASELINE == SF){
			baselineName = "SF";
			TRAINING_INSTANCES_POSITIVE = 800;
		}
		int randSeed;
		randSeed = 0;
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		
		logFileName = Definitions.EXPERIMENTS + "logs/experiments/supervised/SupervisedByOracleSF/" + baselineName + "-" + domain + ".txt";
		try {
			if (!SANDBOX){
				logFile = new File(logFileName);
				pw = new PrintWriter(logFile);
			}
		} catch (FileNotFoundException e1) {
			System.err.printf("Error writing logfile [%s]\n",logFileName);
		}
		
		if (!SANDBOX){
			pw.printf("Domain: %s\n",domain);	
			pw.printf("Experiment#Seed\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
		}else{
			System.out.printf("Domain: %s\n",domain);	
			System.out.printf("Experiment#Seed\t\t\t\t\t|\tPREC\tREC\tF1SCORE\tTP\tTN\tFP\tFN\n");
		}
		for (randSeed = 0; randSeed < 30; randSeed++){
			long startTime = System.currentTimeMillis();
			if (!SANDBOX) System.out.printf("|%s-%s-%d|\n",baselineName,domain,randSeed);
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
			String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domain + ".txt";
			String falsePredictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-false-predictions_" + domain + ".txt";
			MatchesFunctions.restart();
			
			instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
			matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
			Instances allInstances = null;
			Instances allInstances_COMA = null;
			Instances trainingInstances = null;
			Instances testInstances = null;
			
			allInstances_COMA = instancesController.getPoolSet();		//pool set contains COMA extras attributes
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);

			List<Integer> trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> falseInstancesByOracle = new ArrayList<Integer>();
			List<Integer> inconsistentMatches = new ArrayList<Integer>();
			List<Integer> userConfirmedMatches = new ArrayList<Integer>();
			List<Integer> userRejectedMatches = new ArrayList<Integer>();
			List<Integer> networkRejectedMatches = new ArrayList<Integer>();
			List<Integer> networkAcceptedMatches = new ArrayList<Integer>();
			List<Integer> predictionsIDs = new ArrayList<Integer>();
			List<Integer> predictionsOracle = new ArrayList<Integer>();
			
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				predictionsOracle.addAll(trueInstancesByOracle);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
				Functions.requestInstancesFromSimFlood(falseInstancesByOracle, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, falsePredictionsFileName, instancesController);
			}
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Respostas #" + randSeed,pw);
			
			trainingInstances = new Instances(allInstances);	//will contain instances with predicted class values
			testInstances = new Instances(allInstances);		//will be the same as the original set of instances.
			trainingInstances.delete(); 
			
			//--------
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			int qtPositives = trueInstancesByOracle.size();
			int qtNegatives = falseInstancesByOracle.size();

//			System.out.printf("All size: %d (%d)\ntest size: %d (%d)\n",allInstances.numInstances(),Functions.countPositivesInSet(allInstances),testInstances.numInstances(),Functions.countPositivesInSet(testInstances));
			Evaluation eval;
			double precP, recP, f1ScoreP;
			int discP;

			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				eval = new Evaluation(allInstances);
				eval.evaluateModel(randomForest, allInstances);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				discP = (int)eval.numTruePositives(Definitions.TRUE);
				if (!SANDBOX)
					pw.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,randSeed,qtPositives,
							qtNegatives,precP,recP,f1ScoreP,discP);
				else
					System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,randSeed,qtPositives,
						qtNegatives,precP,recP,f1ScoreP,discP);
					
			} catch (Exception e) {
			}
			
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
				pw.printf("----------------------------------------------------------\n");
			}else{
//				System.out.printf("User(+): %d\n" +
//						"User(-): %d\n" +
//						"Rede(+): %d\n" +
//						"Rede(-): %d\n",MatchesFunctions.getUserConfirmedMatches().size(),
//						MatchesFunctions.getUserRejectedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size(),
//						MatchesFunctions.getNetworkRejectedMatches().size());
				System.out.printf("\nExperiment run at: %s\n",c.getTime());
				System.out.printf("Duration (s): %.2f\n", duration/1000.0);
				System.out.printf("----------------------------------------------------------\n");
			}
			
		}
		if (!SANDBOX)
			pw.close();
	}
}
