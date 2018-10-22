package test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class TestAfterCOMALearning {
	static int TRAINING_INSTANCES_POSITIVE = 700;					//default 300
	static int TRAINING_INSTANCES_NEGATIVE = 1000;					//default 1000
	static int NUMBER_OF_TREES_GENERATED = 50;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		int DOMAIN_ID = 3;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
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
			
//			Functions.copyInstancesWithPredictedLabel(trueInstancesByCOMA, allInstances, trainingInstances, Definitions.TRUE);
//			Functions.copyInstancesWithPredictedLabel(falseInstancesByCOMA, allInstances, trainingInstances, Definitions.FALSE);
			
			List<Integer> allTrainingInstances = new ArrayList<Integer>(trueInstancesByCOMA);
			allTrainingInstances.addAll(falseInstancesByCOMA);
			Collections.sort(allTrainingInstances);
			int sizeTraining = allTrainingInstances.size();
			for (int idx = sizeTraining-1; idx > 0; idx--){
				int instID = allTrainingInstances.get(idx);
				allInstances.delete(instID);
			}

			Evaluation eval;
			double precP, recP, f1ScoreP;
			int TP, FP, FN;
			double sumP, sumR, sumF1, sumTP, sumFP, sumFN;
			try {
				allInstances.stratify(10);
				sumP = sumR = sumF1 = sumTP = sumFP = sumFN = 0;
				for (int foldNo = 0; foldNo < 10; foldNo++){
					RandomForest randomForest = new RandomForest();
					randomForest.setSeed(randSeed);
					randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
					trainingInstances = allInstances.testCV(10, foldNo);
					testInstances = allInstances.trainCV(10, foldNo);
					randomForest.buildClassifier(trainingInstances);
					eval = new Evaluation(testInstances);
					eval.evaluateModel(randomForest, testInstances);
					sumP += eval.precision(Definitions.TRUE);
					sumR += eval.recall(Definitions.TRUE);
					sumF1 += eval.fMeasure(Definitions.TRUE);
					sumTP += (int)eval.numTruePositives(Definitions.TRUE);
					sumFP += (int)eval.numFalsePositives(Definitions.TRUE);
					sumFN += (int)eval.numFalseNegatives(Definitions.TRUE);
					System.out.printf(".");
				}
				System.out.printf("\t");
				precP = sumP/10.0;
				recP = sumR/10.0;
				f1ScoreP = sumF1/10.0;
				TP = (int) (sumTP/10);
				FN = (int) (sumFN/10);
				FP = (int) (sumFP/10);
				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\n",domain,randSeed,numPositives,
						numNegatives,precP,recP,f1ScoreP,TP,FP,FN);
			} catch (Exception e) {
				System.err.printf("Error training/testing.\n");
			}
		}
	}

}
