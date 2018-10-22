package experiments.network;

import java.util.ArrayList;
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

public class SupervisedByOracleSFNetwork {

	static int TRAINING_INSTANCES_POSITIVE = 800;
	static int TRAINING_INSTANCES_NEGATIVE = 2000;
	static int NUMBER_OF_TREES_GENERATED = 50;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
//		DOMAIN_ID = Definitions.BUSINESS;
//		DOMAIN_ID = Definitions.MAGAZINE;
//		DOMAIN_ID = Definitions.BOOK;
//		DOMAIN_ID = Definitions.ORDER;
		//options when running from command line
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		
		//initialization
		int randSeed;
//		randSeed = 0;
		
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		System.out.printf("Domain: %s (Training instances only)\n",domain);
		MatchingNetworksInstancesController instancesController;
		MatchesController matchesController;
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + ".arff";
		String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domain + ".txt";
		
		instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
		matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
		Instances allInstances = null;
		Instances trainingInstances = null;
		Instances testInstances = null;
		
		allInstances = instancesController.getPoolSet();		
		Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
		Definitions.FALSE = 1 - Definitions.TRUE;
		trainingInstances = new Instances(allInstances);
		testInstances = new Instances(allInstances);

		List<Integer> trueInstancesBySimFlood = new ArrayList<Integer>();
		List<Integer> falseInstancesBySimFlood = new ArrayList<Integer>();
		Functions.requestInstancesFromSimFlood(trueInstancesBySimFlood, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
//		Functions.requestCorrectLabelInstancesFromSimFlood(trueInstancesBySimFlood, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
		TRAINING_INSTANCES_POSITIVE = trueInstancesBySimFlood.size();
		MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesBySimFlood);
		trueInstancesBySimFlood = new ArrayList<Integer>();
		trueInstancesBySimFlood.addAll(MatchesFunctions.getAcceptedMatches());
		falseInstancesBySimFlood.addAll(MatchesFunctions.getRejectedMatches());
		
		TRAINING_INSTANCES_POSITIVE = trueInstancesBySimFlood.size();
		TRAINING_INSTANCES_NEGATIVE = falseInstancesBySimFlood.size();
		
		for (randSeed = 0; randSeed < 30; randSeed++){
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
		
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesBySimFlood, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesBySimFlood, allInstances, trainingInstances, Definitions.FALSE);
		
			List<Integer> trainingIDs = new ArrayList<Integer>();
			trainingIDs.addAll(trueInstancesBySimFlood);
			trainingIDs.addAll(falseInstancesBySimFlood);
			Functions.removeInstancesFromSet(trainingIDs, testInstances);
			Instances correctLabelTraining = Functions.createSetOfInstances(allInstances, trainingIDs);

			Evaluation eval;
			double precP, recP, f1ScoreP;
			int TP,TN,FP,FN; 
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				eval = new Evaluation(allInstances);
				eval.evaluateModel(randomForest, allInstances);
//				eval = new Evaluation(correctLabelTraining);
//				eval.evaluateModel(randomForest, correctLabelTraining);
//				eval = new Evaluation(testInstances);
//				eval.evaluateModel(randomForest, testInstances);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				TP = (int)eval.numTruePositives(Definitions.TRUE);
				TN = (int)eval.numFalsePositives(Definitions.TRUE);
				FP = (int)eval.numTrueNegatives(Definitions.TRUE);
				FN = (int)eval.numFalseNegatives(Definitions.TRUE);
				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",
						domain,randSeed,TRAINING_INSTANCES_POSITIVE,
						TRAINING_INSTANCES_NEGATIVE,precP,recP,f1ScoreP,TP,TN,FP,FN);
			} catch (Exception e) {
			}
		
//		Evaluation eval;
//		double precP, recP, f1ScoreP;
//		int discP;
//		try {
//			eval = new Evaluation(allInstances);
//			for (int instID = 0; instID < instancesController.getNumInstances(); instID++){
//				Instance inst = allInstances.instance(instID);
//				if (finalAnswers.contains(instID))
//					eval.evaluateModelOnce(Definitions.TRUE, inst);
//				else
//					eval.evaluateModelOnce(Definitions.FALSE, inst);
//			}
//			precP = eval.precision(Definitions.TRUE);
//			recP = eval.recall(Definitions.TRUE);
//			f1ScoreP = eval.fMeasure(Definitions.TRUE);
//			discP = (int)eval.numTruePositives(Definitions.TRUE);
//			System.out.printf("Inconsist. check: %s\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,precP,recP,f1ScoreP,discP);
//		} catch (Exception e) {
//		}
		}
	}

}
