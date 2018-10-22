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

public class RestrictionsAfterCOMA {

	static int TRAINING_INSTANCES_POSITIVE = 700;
	static int TRAINING_INSTANCES_NEGATIVE = 1000;
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
//		options when running from command line
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n> Domain (1-betting/ 2-business/ 3-magazine/ 4-book/ 5-order\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
		}
		//initialization
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		MatchingNetworksInstancesController instancesController;
		MatchesController matchesController;
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";
		instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
		matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
		Instances allInstances = null;
		Instances allInstances_COMA = null;
		
		allInstances_COMA = instancesController.getPoolSet();		//pool set contains COMA extras attributes
		allInstances = new Instances(allInstances_COMA);
		allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
		allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
		Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
		Definitions.FALSE = 1 - Definitions.TRUE;

		List<Integer> trueInstancesByCOMA = new ArrayList<Integer>();
//		Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
		Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, trueInstancesByCOMA, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
		TRAINING_INSTANCES_POSITIVE = trueInstancesByCOMA.size();
		Set<Integer> finalAnswers = MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, trueInstancesByCOMA);
		
		
//		Instances copy = new Instances(allInstances);
//		copy.delete();
//		for (Integer instID: finalAnswers){
//			Instance copyInstance = (Instance) allInstances.instance(instID).copy();
//			if (copyInstance.classValue() == Definitions.FALSE){
//				System.out.printf("%s\n",copyInstance.toString());
//			}
//			copy.add(copyInstance);
//		}
//		System.out.printf("size: %d positives: %d\n",copy.numInstances(),Functions.countPositivesInSet(copy));
//		System.exit(0);
		
		
		
		
		
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
				if (trueInstancesByCOMA.contains(instID))
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
