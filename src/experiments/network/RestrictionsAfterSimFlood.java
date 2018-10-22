package experiments.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class RestrictionsAfterSimFlood {

	static int TRAINING_INSTANCES_POSITIVE = 800;
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
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		MatchingNetworksInstancesController instancesController;
		MatchesController matchesController;
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/MatchingNetwork-" + domain + ".arff";
		String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domain + ".txt";
		
		instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
		matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
		Instances allInstances = null;
		
		allInstances = instancesController.getPoolSet();		
		Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
		Definitions.FALSE = 1 - Definitions.TRUE;

		List<Integer> trueInstancesBySimFlood = new ArrayList<Integer>();
//		Functions.requestInstancesFromSimFlood(trueInstancesBySimFlood, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
		Functions.requestCorrectLabelInstancesFromSimFlood(trueInstancesBySimFlood, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
		TRAINING_INSTANCES_POSITIVE = trueInstancesBySimFlood.size();
		Set<Integer> finalAnswers = MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesBySimFlood);
		
		System.out.printf("%d insert operations\n" +
				"%d inconsistencies of type I found.\n" +
				"%d inconsistencies of type II found.\n",TRAINING_INSTANCES_POSITIVE,
				MatchesFunctions.getNumberOfInconsistenciesOfTypeI(),MatchesFunctions.getNumberOfInconsistenciesOfTypeII());
		
		Evaluation eval;
		double precP, recP, f1ScoreP;
		int discP;
		try {
			eval = new Evaluation(allInstances);
			for (int instID = 0; instID < instancesController.getNumInstances(); instID++){
				Instance inst = allInstances.instance(instID);
				if (finalAnswers.contains(instID))
					eval.evaluateModelOnce(Definitions.TRUE, inst);
				else
					eval.evaluateModelOnce(Definitions.FALSE, inst);
			}
			
			precP = eval.precision(Definitions.TRUE);
			recP = eval.recall(Definitions.TRUE);
			f1ScoreP = eval.fMeasure(Definitions.TRUE);
			discP = (int)eval.numTruePositives(Definitions.TRUE);
			System.out.printf("Inconsist. check: %s\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,precP,recP,f1ScoreP,discP);
		} catch (Exception e) {
		}
		try {
			eval = new Evaluation(allInstances);
			for (int instID = 0; instID < instancesController.getNumInstances(); instID++){
				Instance inst = allInstances.instance(instID);
				if (trueInstancesBySimFlood.contains(instID))
					eval.evaluateModelOnce(Definitions.TRUE, inst);
				else
					eval.evaluateModelOnce(Definitions.FALSE, inst);
			}
			
			precP = eval.precision(Definitions.TRUE);
			recP = eval.recall(Definitions.TRUE);
			f1ScoreP = eval.fMeasure(Definitions.TRUE);
			discP = (int)eval.numTruePositives(Definitions.TRUE);
			System.out.printf("Original Answers: %s\t|\t%.3f\t%.3f\t%.3f\t%d\n",domain,precP,recP,f1ScoreP,discP);
		} catch (Exception e) {
		}
		
	}

}
