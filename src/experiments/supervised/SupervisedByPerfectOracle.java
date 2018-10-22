package experiments.supervised;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class SupervisedByPerfectOracle {
	static int TRAINING_INSTANCES_POSITIVE = 700;					//default 300
	static int TRAINING_INSTANCES_NEGATIVE = 1000;					//default 1000
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
		//options from command line
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		//initialization
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		int randSeed;
		System.out.printf("Domain/Run\tTraining+\tTraining-\t|\tPrec\tRec\tF1Score\tTP\n");
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
		Instances allInstances = null;
		Instances allInstances_COMA = null;
		Instances trainingInstances = null;
		Instances testInstances = null;
		allInstances_COMA = Functions.readInstancesFromARFF(arffFileName);
		allInstances_COMA.deleteAttributeAt(1);								//deleting matchingCandidate attribute
		allInstances_COMA.setClassIndex(allInstances_COMA.numAttributes()-1);	
		for (randSeed = 0; randSeed < 30; randSeed++){
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;

			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			
			List<Integer> trueInstancesByCOMA = new ArrayList<Integer>();
			List<Integer> falseInstancesByCOMA = new ArrayList<Integer>();
			Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			int numPositives = trueInstancesByCOMA.size();
//			TRAINING_INSTANCES_NEGATIVE = numPositives;
			Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, falseInstancesByCOMA, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			int numNegatives = falseInstancesByCOMA.size();
	//		Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, TRAINING_INSTANCES_POSITIVE, falseInstancesByCOMA, TRAINING_INSTANCES_NEGATIVE);
			trainingInstances.delete();
			
			Functions.copyInstancesWithPredictedLabel(trueInstancesByCOMA, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByCOMA, allInstances, trainingInstances, Definitions.FALSE);
			
			
			List<Integer> allTrainingInstances = new ArrayList<Integer>(trueInstancesByCOMA);
			allTrainingInstances.addAll(falseInstancesByCOMA);
			Collections.sort(allTrainingInstances);
			int sizeTraining = allTrainingInstances.size();
			for (int idx = sizeTraining-1; idx > 0; idx--){
				int instID = allTrainingInstances.get(idx);
				testInstances.delete(instID);
			}

			Evaluation eval;
			double precP, recP, f1ScoreP;
			int discP, FP, FN;
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
				FP = (int)eval.numFalsePositives(Definitions.TRUE);
				FN = (int)eval.numFalseNegatives(Definitions.TRUE);
				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\n",domain,randSeed,numPositives,
						numNegatives,precP,recP,f1ScoreP,discP,FP,FN);
			} catch (Exception e) {
			}
		}
	}

}
