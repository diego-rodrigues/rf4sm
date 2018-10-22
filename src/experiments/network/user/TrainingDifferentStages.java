package experiments.network.user;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
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

public class TrainingDifferentStages {

	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 50;
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
		//command java -cp ../lib/wekagp-diego.jar:../bin/ experiments.network.user.TrainingDifferentStages 4 1
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n" +
					"> Domain (1-betting | 2-business | 3-magazine | 4-book | 5-order\n" +
					"> Baseline (1-COMA | 2-Similarity Flooding\n");
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
		logFileName = Definitions.EXPERIMENTS + "logs/experiments/network/user/TrainingDifferentStages/" + baselineName + "-" + domain + ".txt";
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
			System.out.printf("Experiment#Seed\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
		}
		for (randSeed = 0; randSeed < 30; randSeed++){
			if (!SANDBOX) System.out.printf("|%s-%s-%d|\n",baselineName,domain,randSeed);
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
			String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domain + ".txt";
			
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
			List<Integer> uncertainMatches = new ArrayList<Integer>();
			List<Integer> userConfirmedMatches = new ArrayList<Integer>();
			List<Integer> userRejectedMatches = new ArrayList<Integer>();
			List<Integer> networkRejectedMatches = new ArrayList<Integer>();
			
			//Pegando respostas do método automático
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
				int qtInstances = allInstances.numInstances();
				List<Integer> falseInstancesTemp = new ArrayList<Integer>();
				for (int instID = 0; instID < qtInstances; instID++){
					if (!trueInstancesByOracle.contains(instID))
						falseInstancesTemp.add(instID);
				}
				Random r = new Random(randSeed); 
				Collections.shuffle(falseInstancesTemp, r);
				falseInstancesByOracle.addAll(falseInstancesTemp.subList(0, TRAINING_INSTANCES_NEGATIVE));
			}
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Respostas #" + randSeed,pw);
			
			trainingInstances = new Instances(allInstances);	//will contain instances with predicted class values
			testInstances = new Instances(allInstances);		//will be the same as the original set of instances.
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				
				Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas do oraculo)",pw);
			} catch (Exception e) {
			}
			
			//inserindo as respostas do metodo automatico na rede
			MatchesFunctions.insertMatchesNoOrder(matchesController, instancesController, trueInstancesByOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			uncertainMatches.addAll(MatchesFunctions.getInconsistentMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (conservador)",pw);
			
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			
			trainingInstances = new Instances(allInstances);	//will contain instances with predicted class values
			testInstances = new Instances(allInstances);		//will be the same as the original set of instances.
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				
				Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas da rede)",pw);
			} catch (Exception e) {
			}
			
			//usuario corrige respostas inconsistentes entre si
			MatchesFunctions.insertMatchesByAskingUser(matchesController, instancesController, uncertainMatches);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			userConfirmedMatches.addAll(MatchesFunctions.getUserConfirmedMatches());
			userRejectedMatches.addAll(MatchesFunctions.getUserRejectedMatches());
			networkRejectedMatches.addAll(MatchesFunctions.getNetworkRejectedMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de usuário corrigir inconsistências",pw);
			
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			
			trainingInstances = new Instances(allInstances);	//will contain instances with predicted class values
			testInstances = new Instances(allInstances);		//will be the same as the original set of instances.
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				
				Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas da rede + correções do usuário)",pw);
			} catch (Exception e) {
			}
			if (!SANDBOX)
				pw.printf("----------------------------------------------------------\n");
		}
		if (!SANDBOX)
			pw.close();
	}
}
