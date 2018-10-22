package experiments.modules;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.core.Instances;

import experiments.Definitions;
import experiments.functions.Functions;
import experiments.functions.MatchesFunctions;

public class JustPerfectOracle {

	static int TRAINING_INSTANCES_POSITIVE = 600;			//COMA: 600		SF: 800
	static int TRAINING_INSTANCES_NEGATIVE = 2000;			//Default 2000
	static final int COMA = 0;
	static final int SF = 1;
	static int BASELINE = COMA;
	static boolean SANDBOX = true; 
	
	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		
		String domainName;
		String baselineName = "COMA";
		String logFileName;
		File logFile = null;
		PrintWriter pw = null;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
//		DOMAIN_ID = Definitions.BUSINESS;	
//		DOMAIN_ID = Definitions.MAGAZINE;	
//		DOMAIN_ID = Definitions.BOOK;
//		DOMAIN_ID = Definitions.ORDER;
//		BASELINE = SF;
		//--------------------- command line options
		//command java -cp ../lib/wekagp-diego.jar:../bin experiments.network.user.ReconciliationAfterTraining 2 2 &
		if (args.length > 0){
			Definitions.EXPERIMENTS = "../";
			System.out.printf("Args:\n" +
					"> Domain (1-betting | 2-business | 3-magazine | 4-book | 5-order)\n" +
					"> Baseline (1-COMA | 2-Similarity Flooding)\n");
			DOMAIN_ID = Integer.valueOf(args[0]) - 1;
			BASELINE = Integer.valueOf(args[1]) - 1;
			SANDBOX = false;
		}
		//--------------------- 
		//--------------------- initialization
		if (BASELINE == SF){
			baselineName = "SF";
			TRAINING_INSTANCES_POSITIVE = 800;
		}
		for (DOMAIN_ID = 0; DOMAIN_ID < 5; DOMAIN_ID++){
			long startTime = System.currentTimeMillis();
			int randSeed;
			randSeed = 0;
			domainName = Definitions.DOMAIN_LIST[DOMAIN_ID];
			
			MatchingNetworksInstancesController instancesController;
			MatchesController matchesController;
			Instances allInstances = null;
			Instances allInstances_COMA = null;
			//--------------------- 
			//--------------------- Log settings
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/modules/JustOracle/" + baselineName + "-" + domainName +".txt";
			try {
				if (!SANDBOX){
					logFile = new File(logFileName);
					pw = new PrintWriter(logFile);
				}
			} catch (FileNotFoundException e1) {
				System.err.printf("Error writing logfile [%s]\n",logFileName);
			}
			if (!SANDBOX){
				pw.printf("Domain: %s\n",domainName);	
				pw.printf("Experiment#Seed\t|\tPREC\tREC\tF1-SCORE\tTP\tTN\tFP\tFN\n");
			}else{
				System.out.printf("Domain: %s\n",domainName);	
				System.out.printf("Experiment#Seed\t\t\t\t|\tPREC\tREC\tF1SCORE\tTP\tTN\tFP\tFN\n");
			}
			//---------------------
			if (!SANDBOX) System.out.printf("|%s-%s-%d|\n",baselineName,domainName,randSeed);
			String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domainName + "-COMA_AvgMatcher.arff";
			String predictionsFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-predictions_" + domainName + ".txt";
			
			instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
			matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);
			
			allInstances_COMA = instancesController.getPoolSet();		//pool set contains COMA extras attributes
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;

			List<Integer> trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> falseInstancesByOracle = new ArrayList<Integer>();
			List<Integer> inconsistentMatches = new ArrayList<Integer>();
			List<Integer> predictionsOracle = new ArrayList<Integer>();
			
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
				int qtInstances = allInstances.numInstances();
				List<Integer> falseInstancesTemp = new ArrayList<Integer>();
				for (int instID = 0; instID < qtInstances; instID++){
					if (!trueInstancesByOracle.contains(instID))
						falseInstancesTemp.add(instID);
				}
				Random r = new Random(randSeed); 
				Collections.shuffle(falseInstancesTemp, r);
				falseInstancesByOracle.addAll(falseInstancesTemp.subList(0, TRAINING_INSTANCES_NEGATIVE));
			}
			predictionsOracle.addAll(trueInstancesByOracle);
			
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Respostas #" + randSeed,pw);
			
			matchesController.restart();
			MatchesFunctions.insertConsistentMatches(matchesController, instancesController, predictionsOracle);
			trueInstancesByOracle = new ArrayList<Integer>();
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " + rede (no deriving matches)",pw);
			
			inconsistentMatches = new ArrayList<Integer>();
			inconsistentMatches.addAll(MatchesFunctions.getInconsistentMatches());
//			System.out.printf("Inconsist.size(): %d\ntypeI: %d\ntypeII: %d\n",inconsistentMatches.size(),MatchesFunctions.getNumberOfInconsistenciesOfTypeI(),MatchesFunctions.getNumberOfInconsistenciesOfTypeII());
			MatchesFunctions.insertMatchesByAskingUser(matchesController, instancesController, inconsistentMatches);
			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " + rede + usuario",pw);
			
			
			if (!SANDBOX){
				pw.printf("User(+): %d\n" +
					"User(-): %d\n" +
					"Rede(+): %d\n" +
					"Rede(-): %d\n",MatchesFunctions.getUserConfirmedMatches().size(),
					MatchesFunctions.getUserRejectedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size(),
					MatchesFunctions.getNetworkRejectedMatches().size());
				
			}else{
				System.out.printf("User(+): %d\n" +
						"User(-): %d\n" +
						"Rede(+): %d\n" +
						"Rede(-): %d\n",MatchesFunctions.getUserConfirmedMatches().size(),
						MatchesFunctions.getUserRejectedMatches().size(),MatchesFunctions.getNetworkAcceptedMatches().size(),
						MatchesFunctions.getNetworkRejectedMatches().size());
			}
			
			//The perfect oracle (precision = 1)
			trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> correctTrueInstancesByOracle = new ArrayList<Integer>();
			predictionsOracle = new ArrayList<Integer>();
			
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestCorrectLabelInstancesFromCOMA(allInstances_COMA, correctTrueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
			}else{
				Functions.requestCorrectLabelInstancesFromSimFlood(correctTrueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, predictionsFileName, instancesController);
			}
			predictionsOracle.addAll(correctTrueInstancesByOracle);
			
			Functions.measureStatistics(instancesController, predictionsOracle, "Pfct-" + baselineName + " Respostas #" + randSeed,pw);
			
//			matchesController.restart();
//			MatchesFunctions.insertMatchesSequentialWithDerivingMatches(matchesController, instancesController, predictionsOracle);
//			trueInstancesByOracle = new ArrayList<Integer>();
//			trueInstancesByOracle.addAll(MatchesFunctions.getAcceptedMatches());
//			Functions.measureStatistics(instancesController, trueInstancesByOracle, "Pfct-" + baselineName + " Respostas + Rede #" + randSeed,pw);
			
			if (!SANDBOX){
				pw.printf("Original answers size: %d\n" +
						"Perfect answers size: %d\n",trueInstancesByOracle.size(),correctTrueInstancesByOracle.size());
				
				Calendar c = new GregorianCalendar();
				pw.printf("\nExperiment run at: %s\n",c.getTime());
				long duration = System.currentTimeMillis() - startTime;
				pw.printf("Duration (s): %.2f\n", duration/1000.0);
				pw.close();
			}else{
				System.out.printf("Original answers size: %d\n" +
						"Perfect answers size: %d\n",trueInstancesByOracle.size(),correctTrueInstancesByOracle.size());
				System.out.printf("-------------------------------------------------------\n");
			}
		}
	}
}
