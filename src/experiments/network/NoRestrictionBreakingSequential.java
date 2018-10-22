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

public class NoRestrictionBreakingSequential {

	static int TRAINING_INSTANCES_POSITIVE = 600;
	static int TRAINING_INSTANCES_NEGATIVE = 2000;
	static int NUMBER_OF_TREES_GENERATED = 50;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
		DOMAIN_ID = Definitions.BOOK;
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
		System.out.printf("Domain: %s (Test instances only)\n",domain);		
		for (randSeed = 0; randSeed < 30; randSeed++){
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
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
	
			List<Integer> trueInstancesByCOMA = new ArrayList<Integer>();
			List<Integer> falseInstancesByCOMA = new ArrayList<Integer>();
			Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
//			Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
	//		Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByCOMA, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			
			
			//--------------------------- tests ----------------------------
			//book 0[0]-0[1] : 0 alpha
			//book 0[0]-0[2] : 1634 delta
			//book 0[0]-0[3] : 2546 gamma
			//book 0[1]-1[3] : 5137 beta
			
			//System.out.printf("Adding match _alpha_\n");
			//matchesController.forceAddMatch(0, instancesController, true);
			//System.out.printf("-------------------------------------------\n");
			//System.out.printf("Adding match _beta_\n");
			//matchesController.forceAddMatch(5137, instancesController, true);
			//System.out.printf("-------------------------------------------\n");
			//System.out.printf("Adding match _gamma_\n");
			//matchesController.forceAddMatch(2546, instancesController, true);
			//System.out.printf("*******************************************\n");
			//System.out.printf("Test match _alpha_\n");
			//int code = matchesController.canAddMatch(0, instancesController);
			//System.out.printf("code: %d\n",code);
			//System.out.printf("-------------------------------------------\n");
			//System.out.printf("Test match _beta_\n");
			//code = matchesController.canAddMatch(5137, instancesController);
			//System.out.printf("code: %d\n",code);
			//System.out.printf("-------------------------------------------\n");
			//System.out.printf("Test match _gamma_\n");
			//code = matchesController.canAddMatch(2546, instancesController);
			//System.out.printf("code: %d\n",code);
			//System.exit(0);
			//----------------------------------------------------------------
			
			
			
			
			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesByCOMA);
			
			trueInstancesByCOMA = new ArrayList<Integer>();
			trueInstancesByCOMA.addAll(MatchesFunctions.getAcceptedMatches());
			falseInstancesByCOMA.addAll(MatchesFunctions.getRejectedMatches());
			
			int qtPositives = trueInstancesByCOMA.size();
			int qtNegatives = falseInstancesByCOMA.size();
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByCOMA, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByCOMA, allInstances, trainingInstances, Definitions.FALSE);
			
			List<Integer> trainingIDs = new ArrayList<Integer>();
			trainingIDs.addAll(trueInstancesByCOMA);
			trainingIDs.addAll(falseInstancesByCOMA);
			Functions.removeInstancesFromSet(trainingIDs, testInstances);
//			Instances correctLabelTraining = Functions.createSetOfInstances(allInstances, trainingIDs);
			
			Evaluation eval;
			double precP, recP, f1ScoreP;
			int TP,TN,FP,FN; 
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
//				eval = new Evaluation(correctLabelTraining);
//				eval.evaluateModel(randomForest, correctLabelTraining);
				eval = new Evaluation(testInstances);
				eval.evaluateModel(randomForest, testInstances);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				TP = (int)eval.numTruePositives(Definitions.TRUE);
				TN = (int)eval.numFalsePositives(Definitions.TRUE);
				FP = (int)eval.numTrueNegatives(Definitions.TRUE);
				FN = (int)eval.numFalseNegatives(Definitions.TRUE);
				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",
						domain,randSeed,qtPositives,qtNegatives,precP,recP,f1ScoreP,TP,TN,FP,FN);
			} catch (Exception e) {
			}
		}
	}

}
