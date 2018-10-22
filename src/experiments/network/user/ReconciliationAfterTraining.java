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

public class ReconciliationAfterTraining {

	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 20;				//default 50
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
//		BASELINE = SF;
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
		
		logFileName = Definitions.EXPERIMENTS + "logs/experiments/network/user/ReconciliationAfterTraining/" + baselineName + "-" + domain + ".txt";
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
			if (!SANDBOX) System.out.printf("|%s-%s-%d|\n",baselineName,domain,randSeed);
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
			String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domain + ".txt";
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
			
			//--------
			matchesController.restart();
//			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesByOracle);
//			System.out.printf("true size: %d\n",trueInstancesByOracle.size());
			MatchesFunctions.insertConsistentMatches(matchesController, instancesController, trueInstancesByOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
//			System.out.printf("false size: %d\n",falseInstancesByOracle.size());
			//--------
			int sizeT = trueInstancesByOracle.size();
			
			//Making up training set with predicted labels
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			
////			matchesController.restart();
////			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesByOracle);
			MatchesFunctions.insertConsistentMatches(matchesController, instancesController, trueInstancesByOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
//			System.out.printf("true size: %d\n",trueInstancesByOracle.size());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			int sizeF = falseInstancesByOracle.size();
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (baseline + conservador)",pw);
			/*System.out.printf("Accepted: %d\n\tby network: %d\n\tby prediction: %d\n\tby user:%d\n"
					,MatchesFunctions.getAcceptedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size()
					,MatchesFunctions.getAcceptedMatches().size()-MatchesFunctions.getNetworkAcceptedMatches().size()
					,MatchesFunctions.getUserConfirmedMatches().size());
			System.out.printf("Rejected: %d\n\tby network: %d\n\tby user:%d\n"
					,MatchesFunctions.getRejectedMatches().size(),MatchesFunctions.getNetworkRejectedMatches().size()
					,MatchesFunctions.getUserRejectedMatches().size());
			System.out.printf("Inconsistent: %d\n",MatchesFunctions.getInconsistentMatches().size());*/
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
//				trainingInstances.deleteAttributeAt(0);
				randomForest.buildClassifier(trainingInstances);
//				predictionsIDs = Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas do oraculo)",predictionsOracle,pw);
//				System.out.printf("com prioridades: %s\n",predictionsIDs.toString());
				predictionsIDs = Functions.measureStatistics(testInstances, randomForest, "Aval.Total (treinado por respostas do oraculo)",pw);
//				System.out.printf("sem prioridades: %s\n",predictionsIDs.toString());
//				System.exit(0);
			} catch (Exception e) {
			}
			
			//inserindo as respostas do aprendizado na rede (conservador)
//			MatchesFunctions.insertOnlyIfNotBreaksRestrictionsNoOrder(matchesController, instancesController, predictionsIDs);
			
			//inserindo as respostas do aprendizado na rede (ordem)
			matchesController.restart();
//			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, predictionsIDs);
			MatchesFunctions.insertConsistentMatches(matchesController, instancesController,predictionsIDs);
			
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			inconsistentMatches.addAll(MatchesFunctions.getInconsistentMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (conservador)",pw);
//			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de inserir na rede (ordenado)\t",pw);
			
//			System.out.printf("Accepted: %d\n\tby network: %d\n\tby prediction: %d\n\tby user:%d\n"
//					,MatchesFunctions.getAcceptedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size()
//					,MatchesFunctions.getAcceptedMatches().size()-MatchesFunctions.getNetworkAcceptedMatches().size()
//					,MatchesFunctions.getUserConfirmedMatches().size());
//			System.out.printf("Rejected: %d\n\tby network: %d\n\tby user:%d\n"
//					,MatchesFunctions.getRejectedMatches().size(),MatchesFunctions.getNetworkRejectedMatches().size()
//					,MatchesFunctions.getUserRejectedMatches().size());
//			System.out.printf("Inconsistent: %d\n",MatchesFunctions.getInconsistentMatches().size());
			
			//usuario corrige respostas inconsistentes entre si
			MatchesFunctions.insertMatchesByAskingUser(matchesController, instancesController, inconsistentMatches);
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
//			trueInstancesByOracle.addAll(MatchesFunctions.getNetworkAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			userConfirmedMatches.addAll(MatchesFunctions.getUserConfirmedMatches());
			userRejectedMatches.addAll(MatchesFunctions.getUserRejectedMatches());
			networkRejectedMatches.addAll(MatchesFunctions.getNetworkRejectedMatches());
			networkAcceptedMatches.addAll(MatchesFunctions.getNetworkAcceptedMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Depois de usuario corrigir inconsistencias",pw);
			
			
			if (!SANDBOX){
				pw.printf("Treino positivo: %d\n" +
						"Treino negativo: %d\n" +
						"Treino total: %d\n", sizeT, sizeF, sizeT + sizeF);
				pw.printf("Correcoes do usuario (+): %d\n" +
						"Correcoes do usuario (-): %d\n" +
						"Confirmacoes pelas regras de rede: %d\n" +
						"Rejeicoes pelas regras de rede: %d\n",userConfirmedMatches.size(),
						userRejectedMatches.size(),networkAcceptedMatches.size(),networkRejectedMatches.size());
			}else
				System.out.printf("Correcoes do usuario (+): %d\n" +
					"Correcoes do usuario (-): %d\n" +
					"Confirmacoes pelas regras de rede: %d\n" +
					"Rejeicoes pelas regras de rede: %d\n",userConfirmedMatches.size(),
					userRejectedMatches.size(),networkAcceptedMatches.size(),networkRejectedMatches.size());
			
			if (!SANDBOX)
				pw.printf("----------------------------------------------------------\n");
			else
				System.out.printf("----------------------------------------------------------\n");
		}
		if (!SANDBOX)
			pw.close();
	}
}
