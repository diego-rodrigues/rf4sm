package experiments.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import structures.schema.InstancesController;
import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class NoRestrictionBreakingNoOrder {

	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 20;				//defaul 50
	static final int COMA = 0;
	static final int SF = 1;
	static int BASELINE = COMA;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		String baselineName = "COMA";
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
		DOMAIN_ID = Definitions.BOOK;
		DOMAIN_ID = Definitions.ORDER;
		BASELINE = SF;
		//options when running from command line
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		
		//initialization
		int randSeed;
		if (BASELINE == SF){
			baselineName = "SF";
			TRAINING_INSTANCES_POSITIVE = 800;
		}
			
//		randSeed = 0;
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		System.out.printf("Domain: %s\n",domain);	
		System.out.printf("Experiment#Seed\tTr+\tTr-\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
		for (randSeed = 0; randSeed < 1; randSeed++){
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
			if (BASELINE == COMA)
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			else
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
			
//			Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Answers #"+randSeed, null);
			matchesController.restart();
//			MatchesFunctions.insertMatchesSequentialWithNoDerivingMatches(matchesController, instancesController, trueInstancesByOracle);
			MatchesFunctions.insertMatchesNoOrder(matchesController, instancesController, trueInstancesByOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			
			int qtPositives = trueInstancesByOracle.size();
			int qtNegatives = falseInstancesByOracle.size();
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			
			MatchesFunctions.insertConsistentMatches(matchesController, instancesController, trueInstancesByOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
//			System.out.printf("true size: %d\n",trueInstancesByOracle.size());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			
			List<Integer> trainingIDs = new ArrayList<Integer>();
			trainingIDs.addAll(trueInstancesByOracle);
			trainingIDs.addAll(falseInstancesByOracle);
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Rede de respostas (conservador)", null);
			Functions.removeInstancesFromSet(trainingIDs, testInstances);
			Instances correctLabeledTraining = Functions.createSetOfInstances(allInstances, trainingIDs);
			
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				
				Functions.measureStatistics(correctLabeledTraining, randomForest, "Aval treino",null);
				Functions.measureStatistics(testInstances, randomForest, "Aval teste",null);
				Functions.measureStatistics(allInstances, randomForest, "Aval todas as inst",null);
				
				int pos = Functions.countPositivesInSet(allInstances);
				int numInstances = allInstances.numInstances();
				int posTraining = Functions.countPositivesInSet(correctLabeledTraining);
				int numInstancesTraining = correctLabeledTraining.numInstances();
				int posTest = Functions.countPositivesInSet(testInstances);
				int numInstancesTest = testInstances.numInstances();
				System.out.printf("\n");
				System.out.printf("Tamanho do treino:\t%d\t(+)\t%d\t(-)\n",trueInstancesByOracle.size(),falseInstancesByOracle.size());
				System.out.printf("Tarefa:\t#positivos\t#instancias\n" +
						"%s\t%d\t%d\n" +
						"Aval treino\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n" +
						"Aval teste\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n",baselineName,pos,numInstances,
						posTraining,numInstancesTraining,posTraining*100.0/pos,numInstancesTraining*100.0/numInstances,
						posTest,numInstancesTest,posTest*100.0/pos,numInstancesTest*100.0/numInstances);
			} catch (Exception e) {
			}
			System.out.printf("-----------------------------------------------------\n");
		}
	}
}