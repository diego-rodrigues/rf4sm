package experiments.modules;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;

import structures.schema.MatchingNetworksInstancesController;
import weka.core.Instances;
import experiments.Definitions;
import experiments.functions.Functions;

public class OraclePrecision {
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
			Instances allInstances = null;
			Instances allInstances_COMA = null;
			//--------------------- 
			//--------------------- Log settings
			logFileName = Definitions.EXPERIMENTS + "logs/experiments/modules/OraclePrecision/" + baselineName + "-" + domainName +".txt";
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
			String truePredictionsBySFFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-true-predictions_" + domainName + ".txt";
			String falsePredictionsBySFFileName = Definitions.EXPERIMENTS + "docs/Predictions_SimFlood/SimFlood-false-predictions_" + domainName + ".txt";
			
			instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
			
			allInstances_COMA = instancesController.getPoolSet();		//pool set contains COMA extras attributes
			allInstances = new Instances(allInstances_COMA);
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing COMADecisionMatcher attribute
			allInstances.deleteAttributeAt(allInstances.numAttributes()-2);		//removing comaMatchersAverage attribute
			Definitions.TRUE = allInstances.classAttribute().indexOfValue("true");
			Definitions.FALSE = 1 - Definitions.TRUE;

			List<Integer> trueInstancesByOracle = new ArrayList<Integer>();
			List<Integer> falseInstancesByOracle = new ArrayList<Integer>();
			List<Integer> predictionsOracle = new ArrayList<Integer>();
			
			TRAINING_INSTANCES_NEGATIVE = 4000;
//			TRAINING_INSTANCES_NEGATIVE = allInstances.numInstances() / 10;
//			if (TRAINING_INSTANCES_NEGATIVE < 2000) TRAINING_INSTANCES_NEGATIVE = 2000; 
			
			//Pegando respostas do método automatico
			if (BASELINE == COMA){
				Functions.requestInstancesFromCOMA(allInstances_COMA, trueInstancesByOracle, Definitions.TRUE, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING);
				Functions.requestInstancesFromCOMA(allInstances_COMA, falseInstancesByOracle, Definitions.FALSE, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING);
			}else{
				Functions.requestInstancesFromSimFlood(trueInstancesByOracle, TRAINING_INSTANCES_POSITIVE, Functions.DESCENDING, truePredictionsBySFFileName, instancesController);
				Functions.requestInstancesFromSimFlood(falseInstancesByOracle, TRAINING_INSTANCES_NEGATIVE, Functions.DESCENDING, falsePredictionsBySFFileName, instancesController);
//				int qtInstances = allInstances.numInstances();
//				List<Integer> falseInstancesTemp = new ArrayList<Integer>();
//				for (int instID = 0; instID < qtInstances; instID++){
//					if (!trueInstancesByOracle.contains(instID))
//						falseInstancesTemp.add(instID);
//				}
//				Random r = new Random(randSeed); 
//				Collections.shuffle(falseInstancesTemp, r);
//				falseInstancesByOracle.addAll(falseInstancesTemp.subList(0, TRAINING_INSTANCES_NEGATIVE));
			}
//			predictionsOracle.addAll(trueInstancesByOracle);
			
			predictionsOracle.addAll(falseInstancesByOracle);
			
//			int numberPositives = Functions.countPositivesInSet(allInstances);
//			int tpTrain = 0;
//			for (int i=0; i<trueInstancesByOracle.size(); i++){
//				int instID = trueInstancesByOracle.get(i);
//				int classVal = instancesController.getInstanceClassValue(instID);
//				if (classVal == Definitions.TRUE) tpTrain++;
//			}
//			System.out.printf("Number of positives total: %d\n" +
//					"Number of positives in training: %d\n" +
//					"Number of positives remaining: %d (%.2f%%)\n",numberPositives, tpTrain, numberPositives-tpTrain, 
//					((numberPositives-tpTrain)*100)/(double)(numberPositives));
//			if (DOMAIN_ID >= 0) continue;
			
			
			System.out.printf("Desempenho das respostas do oráculo...\n");
			int cor = 0;
			int t = 0;
//			int goal = 5;
			for (int i=0; i<predictionsOracle.size(); i++){
				int instID = predictionsOracle.get(i);
				int classVal = instancesController.getInstanceClassValue(instID);
				t++;
				if (classVal == Definitions.TRUE) cor++;
//				if (((i+1)*100)/(double)predictionsOracle.size() >= goal){
//					goal += 5;
//					System.out.printf("%.2f%% - %.2f\n",((i+1)*100)/(double)predictionsOracle.size(),cor/(double)t);
//				}
				if (i%10 == 0){
					System.out.printf("%4d - %4d - %.2f\t||",i,cor,cor/(double)t);
				}
				if (i%60 == 0){
					System.out.printf("\n",i,cor);
				}
			}
			System.out.printf("---------------------------------------\n");
			System.exit(0);
			
			
			//janela
			int sizeWindow = (int)(predictionsOracle.size()*0.25);  //10%
			int numCor = 0;
			int maxBound = 0;
			for (maxBound=0; maxBound<sizeWindow; maxBound++){
				int instID = predictionsOracle.get(maxBound);
				int classVal = instancesController.getInstanceClassValue(instID);
				if (classVal == Definitions.TRUE) numCor++;
			}
			int lowBound = maxBound - sizeWindow;
			double prec = numCor/(double)sizeWindow;
			System.out.printf("De %d ate %d: %.2f [%d / %d]\n",lowBound,maxBound-1,prec,numCor,sizeWindow);
			do{
				//tira o primeiro
				int instID = predictionsOracle.get(lowBound);
				int classVal = instancesController.getInstanceClassValue(instID);
				if (classVal == Definitions.TRUE) numCor--;
				//insere o ultimo
				instID = predictionsOracle.get(maxBound);
				classVal = instancesController.getInstanceClassValue(instID);
				if (classVal == Definitions.TRUE) numCor++;
				prec = numCor/(double)sizeWindow;
				lowBound++;
				System.out.printf("De %d ate %d: %.2f [%d / %d] (%.2f%%)\n",lowBound,maxBound,prec,numCor,sizeWindow,(100.0*(maxBound+1))/(double)(predictionsOracle.size()-1.0));
				maxBound++;
			}while (maxBound+1 < predictionsOracle.size());
			//fim janela
			
			Functions.measureStatistics(instancesController, trueInstancesByOracle, baselineName + " Respostas #" + randSeed,pw);
			Calendar c = new GregorianCalendar();
			long duration = System.currentTimeMillis() - startTime;
			if (!SANDBOX){
				pw.printf("\nExperiment run at: %s\n",c.getTime());
				pw.printf("Duration (s): %.2f\n", duration/1000.0);
				pw.close();
			}else{
				System.out.printf("\nExperiment run at: %s\n",c.getTime());
				System.out.printf("Duration (s): %.2f\n", duration/1000.0);
				System.out.printf("\n");
			}
			System.exit(0);
		}
	}
}