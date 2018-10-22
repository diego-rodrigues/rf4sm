package experiments.supervised;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class SupervisedByOracle {
	static int TRAINING_INSTANCES_POSITIVE = 600;					//default 300/600
	static int TRAINING_INSTANCES_NEGATIVE = 2000;					//default 1000/2000
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
		System.out.printf("Domain/Run\tTraining+\tTraining-\t|\tPrec\tRec\tF1Score\tTP\n");
		for (randSeed = 0; randSeed < 1; randSeed++){
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
			Instances allInstances = null;
			Instances allInstances_COMA = null;
			Instances trainingInstances = null;
			Instances testInstances = null;
			allInstances_COMA = Functions.readInstancesFromARFF(arffFileName);
			allInstances_COMA.deleteAttributeAt(1);								//deleting matchingCandidate attribute
			allInstances_COMA.setClassIndex(allInstances_COMA.numAttributes()-1);	
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
			int qtPositives = trueInstancesByCOMA.size();
			Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByCOMA, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			int qtNegatives = falseInstancesByCOMA.size();
	//		Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, TRAINING_INSTANCES_POSITIVE, falseInstancesByCOMA, TRAINING_INSTANCES_NEGATIVE);
			trainingInstances.delete(); 
			Functions.copyInstancesWithPredictedLabel(trueInstancesByCOMA, allInstances, trainingInstances, Definitions.TRUE);
			Functions.copyInstancesWithPredictedLabel(falseInstancesByCOMA, allInstances, trainingInstances, Definitions.FALSE);

//			System.out.printf("All size: %d (%d)\ntest size: %d (%d)\n",allInstances.numInstances(),Functions.countPositivesInSet(allInstances),testInstances.numInstances(),Functions.countPositivesInSet(testInstances));
			Evaluation eval;
			double precP, recP, f1ScoreP;
			int discP;

//			try {
//				eval = new Evaluation(allInstances);
//				for (int instID = 0; instID < allInstances.numInstances(); instID++){
//					Instance inst = allInstances.instance(instID);
//					if (trueInstancesByCOMA.contains(instID))
//						eval.evaluateModelOnce(Definitions.TRUE, inst);
//					else
//						eval.evaluateModelOnce(Definitions.FALSE, inst);
//				}
//				
//				precP = eval.precision(Definitions.TRUE);
//				recP = eval.recall(Definitions.TRUE);
//				f1ScoreP = eval.fMeasure(Definitions.TRUE);
//				discP = (int)eval.numTruePositives(Definitions.TRUE);
//				System.out.printf("Original Answers: %s\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,precP,recP,f1ScoreP,discP);
//			} catch (Exception e) {
//			}
			
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
				System.out.printf("%s#%02d\t%5d\t%5d\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,randSeed,qtPositives,
						qtNegatives,precP,recP,f1ScoreP,discP);
			} catch (Exception e) {
			}
//			System.exit(0);
		}
	}

}
