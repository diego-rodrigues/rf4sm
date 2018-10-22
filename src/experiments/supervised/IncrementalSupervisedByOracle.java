package experiments.supervised;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class IncrementalSupervisedByOracle {
	static int TRAINING_INSTANCES_POSITIVE = 50;					//default 300
	static int TRAINING_INSTANCES_NEGATIVE = 1000;					//default 1000
	static int NUMBER_OF_TREES_GENERATED = 50;
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
		DOMAIN_ID = Definitions.BOOK;
		DOMAIN_ID = Definitions.ORDER;
		//options from command line
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		//initialization
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		int randSeed;
//		System.out.printf("Domain/Run\tTraining+\tTraining-\t|\tPrec\tRec\tF1Score\tTP\n");
		System.out.printf("Domain: %s\n",domain);
		randSeed = 0;
		TRAINING_INSTANCES_POSITIVE = 00;
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
		Instances allInstances = null;
		Instances allInstances_COMA = null;
		Instances trainingInstances = null;
		Instances testInstances = null;
		allInstances_COMA = Functions.readInstancesFromARFF(arffFileName);
		allInstances_COMA.deleteAttributeAt(1);								//deleting matchingCandidate attribute
		allInstances_COMA.setClassIndex(allInstances_COMA.numAttributes()-1);	
		
		int lastRoundPositives = -10;
		int maxPositivesReturnedByOracle = 0;
		while (TRAINING_INSTANCES_POSITIVE != lastRoundPositives){
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;
			lastRoundPositives = TRAINING_INSTANCES_POSITIVE;
			TRAINING_INSTANCES_POSITIVE += 5; 
	
			trainingInstances = new Instances(allInstances);
			testInstances = new Instances(allInstances);
			
			List<Integer> trueInstancesByCOMA = new ArrayList<Integer>();
			List<Integer> falseInstancesByCOMA = new ArrayList<Integer>();
			if (TRAINING_INSTANCES_POSITIVE == 5){			//if it is the first round
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, 100000, Functions.DESCENDING);
				maxPositivesReturnedByOracle = trueInstancesByCOMA.size();
				trueInstancesByCOMA = new ArrayList<Integer>();
			}
			Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			TRAINING_INSTANCES_POSITIVE = trueInstancesByCOMA.size();
			Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByCOMA, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
	//		Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, TRAINING_INSTANCES_POSITIVE, falseInstancesByCOMA, TRAINING_INSTANCES_NEGATIVE);
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByCOMA, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByCOMA, allInstances, trainingInstances, Definitions.FALSE);

			Evaluation eval;
			double precP, recP, f1ScoreP;
			int discP;
			try {
				RandomForest randomForest = new RandomForest();
				randomForest.setSeed(randSeed);
				randomForest.setNumTrees(NUMBER_OF_TREES_GENERATED);
				randomForest.buildClassifier(trainingInstances);
				eval = new Evaluation(testInstances);
				eval.evaluateModel(randomForest, testInstances);
				precP = eval.precision(Definitions.TRUE);
				recP = eval.recall(Definitions.TRUE);
				f1ScoreP = eval.fMeasure(Definitions.TRUE);
				discP = (int)eval.numTruePositives(Definitions.TRUE);
//				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,randSeed,TRAINING_INSTANCES_POSITIVE,
//						TRAINING_INSTANCES_NEGATIVE,precP,recP,f1ScoreP,discP);
				double perc = TRAINING_INSTANCES_POSITIVE*100/(double)(maxPositivesReturnedByOracle);
				System.out.printf("%.2f\t%d\t%.3f\n",perc,TRAINING_INSTANCES_POSITIVE,f1ScoreP);
			} catch (Exception e) {
			}
		}
	}

}
