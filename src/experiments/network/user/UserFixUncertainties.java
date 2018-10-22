package experiments.network.user;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
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

public class UserFixUncertainties {

	static int TRAINING_INSTANCES_POSITIVE = 800;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static int NUMBER_OF_TREES_GENERATED = 50;
	static final int COMA = 0;
	static final int SF = 1;
	static int BASELINE = COMA;
	static final boolean SANDBOX = true; 
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		String baselineName = "COMA";
		String logFileName;
		File logFile = null;
		PrintWriter pw = null;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
		DOMAIN_ID = Definitions.BOOK;
		DOMAIN_ID = Definitions.ORDER;
		BASELINE = SF;
		//options when running from command line
		//command java -cp ../lib/wekagp-diego.jar:../bin/ experiments.network.user.UserFixUncertainties 4 1
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n" +
					"> Domain (1-betting | 2-business | 3-magazine | 4-book | 5-order\n" +
					"> Baseline (1-COMA | 2-Similarity Flooding\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			BASELINE = Integer.valueOf(args[1]) - 1;
		}
		
		//initialization
		if (BASELINE == SF) baselineName = "SF";
		int randSeed;
//		randSeed = 0;
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		if (!SANDBOX)
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/network/user/UserFixUncertainties/" + baselineName + "-" + domain + ".txt";
		else
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/sandbox-results.txt";
		try {
			logFile = new File(logFileName);
			pw = new PrintWriter(logFile);
		} catch (FileNotFoundException e1) {
			System.err.printf("Error writing logfile [%s]\n",logFileName);
		}
		
//		System.out.printf("Domain: %s\n",domain);	
//		System.out.printf("Experiment#Seed\tTr+\tTr-\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
		pw.printf("Domain: %s\n",domain);	
		pw.printf("Experiment#Seed\tTr+\tTr-\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
		for (randSeed = 0; randSeed < 30; randSeed++){
			System.out.printf("|%s-%s-%d|\t",baselineName,domain,randSeed);
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
			
			if (BASELINE == COMA)
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			else
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
			
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Answers #" + randSeed,pw);
			MatchesFunctions.insertMatchesNoOrder(matchesController, instancesController, trueInstancesByOracle);
			
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			uncertainMatches.addAll(MatchesFunctions.getInconsistentMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "After Insert with no Restrictions (conservative)",pw);
			
			MatchesFunctions.insertMatchesByAskingUser(matchesController, instancesController, uncertainMatches);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByOracle = new ArrayList<Integer>();
			falseInstancesByOracle.addAll(MatchesFunctions.getRejectedMatches());
			userConfirmedMatches.addAll(MatchesFunctions.getUserConfirmedMatches());
			userRejectedMatches.addAll(MatchesFunctions.getUserRejectedMatches());
			networkRejectedMatches.addAll(MatchesFunctions.getNetworkRejectedMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, "After User-check of Uncertainties",pw);
			
			int qtPositives = trueInstancesByOracle.size();
			int qtNegatives = falseInstancesByOracle.size();
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByOracle, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByOracle, allInstances, trainingInstances, Definitions.FALSE);
			
			List<Integer> trainingIDs = new ArrayList<Integer>();
			trainingIDs.addAll(trueInstancesByOracle);
			trainingIDs.addAll(falseInstancesByOracle);
			Functions.removeInstancesFromSet(trainingIDs, testInstances);
			Instances correctLabeledTraining = Functions.createSetOfInstances(allInstances, trainingIDs);
			
			
			
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				
				Functions.measureStatistics(correctLabeledTraining, randomForest, "Eval training set",pw);
				Functions.measureStatistics(testInstances, randomForest, "Eval test set",pw);
				Functions.measureStatistics(allInstances, randomForest, "Eval all instances",pw);
				
				int pos = Functions.countPositivesInSet(allInstances);
				int numInstances = allInstances.numInstances();
				int posTraining = Functions.countPositivesInSet(correctLabeledTraining);
				int numInstancesTraining = correctLabeledTraining.numInstances();
				int posTest = Functions.countPositivesInSet(testInstances);
				int numInstancesTest = testInstances.numInstances();
//				System.out.printf("\n");
//				System.out.printf("User interventions:\n" +
//						"Total: %d\n" +
//						"Positive: %d\n" +
//						"Negative: %d\n" +
//						"Network rejections: %d\n",userConfirmedMatches.size() + userRejectedMatches.size(),userConfirmedMatches.size(),
//						userRejectedMatches.size(),networkRejectedMatches.size());
//				System.out.printf("Training size:\t%d\t(+)\t%d\t(-)\n",trueInstancesByOracle.size(),falseInstancesByOracle.size());
//				System.out.printf("Size of task:\t#positives\t#instances\n" +
//						"%s\t%d\t%d\n" +
//						"Eval training\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n" +
//						"Eval test\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n",baselineName,pos,numInstances,
//						posTraining,numInstancesTraining,posTraining*100.0/pos,numInstancesTraining*100.0/numInstances,
//						posTest,numInstancesTest,posTest*100.0/pos,numInstancesTest*100.0/numInstances);
				pw.printf("\n");
				pw.printf("User interventions:\n" +
						"Total: %d\n" +
						"Positive: %d\n" +
						"Negative: %d\n" +
						"Network rejections: %d\n",userConfirmedMatches.size() + userRejectedMatches.size(),userConfirmedMatches.size(),
						userRejectedMatches.size(),networkRejectedMatches.size());
				pw.printf("Training size:\t%d\t(+)\t%d\t(-)\n",trueInstancesByOracle.size(),falseInstancesByOracle.size());
				pw.printf("Size of task:\t#positives\t#instances\n" +
						"%s\t%d\t%d\n" +
						"Eval training\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n" +
						"Eval test\t%d\t%d\t(%.2f%%)\t(%.2f%%)\n",baselineName,pos,numInstances,
						posTraining,numInstancesTraining,posTraining*100.0/pos,numInstancesTraining*100.0/numInstances,
						posTest,numInstancesTest,posTest*100.0/pos,numInstancesTest*100.0/numInstances);
			} catch (Exception e) {
			}
//			System.out.printf("-----------------------------------------------------\n");
			pw.printf("-----------------------------------------------------\n");
		}
		pw.close();
		
	}
	
	
}
