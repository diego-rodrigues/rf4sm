package experiments.functions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import structures.schema.InstancesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import experiments.Definitions;

public class Functions {
	
	public static final int F1SCORE = 1;
	public static final int F2SCORE = 2;
	
	public static final int DESCENDING = 1;
	public static final int ASCENDING = 0;
	
	private static class Contender{
		int index;
		double score;
		double score2;
		
		public Contender(int index, double score){
			this.index = index;
			this.score = score;
			this.score2 = 1;
		}
		
		public Contender(int index, double score, double score2){
			this.index = index;
			this.score = score;
			this.score2 = score2;
		}
	}
	
	/**
	 * Use this comparator to get descending ordering.
	 */
	private static Comparator<Contender> descendingComparator = new Comparator<Contender>() {
		@Override
		public int compare(Contender c1, Contender c2) {
			if (c1.score == c2.score){
				if (c1.score2 == c2.score2)
					return 0;
				else{
					if (c1.score2 > c2.score2) return -1;
					else return 1;
				}
			}else
				if (c1.score > c2.score) return -1;
				else return 1;
		}
	};
	
	/**
	 * Use this comparator to get descending ordering.
	 */
	private static Comparator<Contender> ascendingComparator = new Comparator<Contender>() {
		@Override
		public int compare(Contender c1, Contender c2) {
			if (c1.score == c2.score){
				if (c1.score2 == c2.score2)
					return 0;
				else{
					if (c1.score2 < c2.score2) return -1;
					else return 1;
				}
			}else
				if (c1.score < c2.score) return -1;
				else return 1;
		}
	};
	

	/**
	 * Function to rank scores, it return the indexes ranked from highest score to lowest score
	 * @param scores the numbers to be ranked
	 * @param numRanked the size of the ranked array returned
	 * @return an array containing the indexes of scores ranked from highest to lowest
	 */
	public static int[] rankingScores(double[] scores, int numRanked){
		int numContenders = scores.length;
		Contender[] contenders = new Contender[numContenders];
		for (int i = 0; i < numContenders; i++)
			contenders[i] = new Contender(i, scores[i]);
		Arrays.sort(contenders, descendingComparator);
		int[] ranked = new int[numRanked];
		for (int i = 0; (i < numRanked) && (i < numContenders); i++)
			ranked[i] = contenders[i].index;
		return ranked;
	}

	/**
	 * Decides which are the best trees among the given according to a deciding function. 
	 * @param DECIDER Use Functions.F1SCORE to decide using F1-score. Use Functions.F2SCORE to decide using F2-score.
	 */
	public static Classifier[] decideBestTrees(Classifier[] trees, Instances set, int topK, int DECIDER){
		double[] scores = null;
		switch (DECIDER){
		case F1SCORE:{
			scores = evaluateTreesByF1(trees, set);
			break;
		}case F2SCORE:{
			scores = evaluateTreesByF2(trees, set);
			break;
		}default:{
			System.err.printf("Error choosing deciding function to rank trees.\n");
		}
		}
		return decideBestTrees(trees, topK, scores);
	}
	
	/**
	 * Decides which are the best trees among the given according to given scores. 
	 */
	public static Classifier[] decideBestTrees(Classifier[] trees, int topK, double[] scores){
		int[] rankedIndexes = null;
		Classifier[] bestTrees = new Classifier[topK];
		rankedIndexes = Functions.rankingScores(scores, topK);
		int rankNo;
		for (rankNo = 0; rankNo < topK; rankNo++){
			int treeID = rankedIndexes[rankNo];
			try {
				bestTrees[rankNo] = Classifier.makeCopy(trees[treeID]);
			} catch (Exception e) {
				System.out.printf("Error copying trees when deciding best trees.\n");
			}
		}
		return bestTrees;
	}
	
	/**
	 * Copy the K first trees of the array.  
	 */
	public static Classifier[] copyKTrees(Classifier[] trees, int topK){
		Classifier[] copyTrees = new Classifier[topK];
		int idTree;
		for (idTree = 0; idTree < topK; idTree++){
			try {
				copyTrees[idTree] = Classifier.makeCopy(trees[idTree]);
			} catch (Exception e) {
				System.out.printf("Error copying trees.\n");
			}
		}
		return copyTrees;
	}
	
	/**
	 * Copy the trees that its score is higher or equal than the specified threshold 
	 */
	public static Classifier[] copyTreesAboveThreshold(Classifier[] trees, double[] scores, double threshold){
		List<Integer> ids = new ArrayList<Integer>();
		int maxTrees = trees.length;
		for (int idTree = 0; idTree < maxTrees; idTree++)
			if (scores[idTree] >= threshold) ids.add(idTree);

		Classifier[] copyTrees = new Classifier[ids.size()];
		int copyID = 0;
		for (int idTree: ids){
			try {
				copyTrees[copyID] = Classifier.makeCopy(trees[idTree]); 
				copyID++;
			} catch (Exception e) {
				System.out.printf("Error copying trees.\n");
			}
		}
		return copyTrees;
	}
	
	/**
	 * Evaluates trees by F1Score. Trees scores are returned in an array of doubles.
	 */
	private static double[] evaluateTreesByF1(Classifier[] trees, Instances set){
		int NUMBER_OF_TREES = trees.length;
		Definitions.TRUE = set.classAttribute().indexOfValue("true");
		double[] scores = new double[NUMBER_OF_TREES];
		for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
			Evaluation ev;
			try {
				ev = new Evaluation(set);
				ev.evaluateModel(trees[treeID], set);
				scores[treeID] = ev.fMeasure(Definitions.TRUE);
			} catch (Exception e) {
				System.err.printf("Error evaluating set of instances by F1.\n");
			}
		}
		return scores;
	}
	
	/**
	 * Evaluates trees by F2Score. Trees scores are returned in an array of doubles.
	 */
	private static double[] evaluateTreesByF2(Classifier[] trees, Instances set){
		int NUMBER_OF_TREES = trees.length;
		Definitions.TRUE = set.classAttribute().indexOfValue("true");
		int tp, tn, fp, fn;
		double[] scores = new double[NUMBER_OF_TREES];
		for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
			Evaluation ev;
			try {
				ev = new Evaluation(set);
				ev.evaluateModel(trees[treeID], set);
				tp = (int) ev.numTruePositives(Definitions.TRUE);
				fp = (int) ev.numFalsePositives(Definitions.TRUE);
				tn = (int) ev.numTrueNegatives(Definitions.TRUE);
				fn = (int) ev.numFalseNegatives(Definitions.TRUE);
				scores[treeID] = fScore(tp, tn, fp, fn, 2);
			} catch (Exception e) {
				System.err.printf("Error evaluating set of instances by F1.\n");
			}
		}
		return scores;
	}
	
	/**
	 * Function to calculate F1 and F2 scores simultaneously. It stores the results in f1Scores/f2Scores arrays. 
	 */
	public static void calcFScores(Classifier[] trees, Instances set, double[] f1Scores, double[] f2Scores){
		int NUMBER_OF_TREES = trees.length;
		Definitions.TRUE = set.classAttribute().indexOfValue("true");
		int tp, tn, fp, fn;
		for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
			Evaluation ev;
			try {
				ev = new Evaluation(set);
				ev.evaluateModel(trees[treeID], set);
				tp = (int) ev.numTruePositives(Definitions.TRUE);
				fp = (int) ev.numFalsePositives(Definitions.TRUE);
				tn = (int) ev.numTrueNegatives(Definitions.TRUE);
				fn = (int) ev.numFalseNegatives(Definitions.TRUE);
				f1Scores[treeID] = fScore(tp, tn, fp, fn, 1);
				f2Scores[treeID] = fScore(tp, tn, fp, fn, 2);
			} catch (Exception e) {
				System.err.printf("Error evaluating set of instances by F1/F2.\n");
			}
		}
	}
	
	/**
	 * Function to calculate F1, F2, Accuracy and Out-of-bag-error scores simultaneously. 
	 * It stores the results in f1Scores,f2Scores,accuracy arrays. 
	 */
	public static void calcStatisticFunctions(Classifier[] trees, Instances set, double[] f1Scores, double[] f2Scores,
			double[] accuracyScores){
		int NUMBER_OF_TREES = trees.length;
		Definitions.TRUE = set.classAttribute().indexOfValue("true");
		int tp, tn, fp, fn;
		for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
			Evaluation ev;
			try {
				ev = new Evaluation(set);
				ev.evaluateModel(trees[treeID], set);
				tp = (int) ev.numTruePositives(Definitions.TRUE);
				fp = (int) ev.numFalsePositives(Definitions.TRUE);
				tn = (int) ev.numTrueNegatives(Definitions.TRUE);
				fn = (int) ev.numFalseNegatives(Definitions.TRUE);
				f1Scores[treeID] = fScore(tp, tn, fp, fn, 1);
				f2Scores[treeID] = fScore(tp, tn, fp, fn, 2);
				accuracyScores[treeID] = ev.pctCorrect()/100;
			} catch (Exception e) {
				System.err.printf("Error evaluating set of instances by F1,F2,Accuracy.\n");
			}
		}
	}
	
	/**
	 * Calculate FScores for a set of trees.
	 */
	public static void calcFScores(Classifier[] trees, Instances set, double[] fScores, double beta){
		int NUMBER_OF_TREES = trees.length;
		Definitions.TRUE = set.classAttribute().indexOfValue("true");
		int tp, tn, fp, fn;
		for (int treeID = 0; treeID < NUMBER_OF_TREES; treeID++){
			Evaluation ev;
			try {
				ev = new Evaluation(set);
				ev.evaluateModel(trees[treeID], set);
				tp = (int) ev.numTruePositives(Definitions.TRUE);
				fp = (int) ev.numFalsePositives(Definitions.TRUE);
				tn = (int) ev.numTrueNegatives(Definitions.TRUE);
				fn = (int) ev.numFalseNegatives(Definitions.TRUE);
				fScores[treeID] = fScore(tp, tn, fp, fn, beta);
			} catch (Exception e) {
				System.err.printf("Error evaluating set of instances by F1,F2,Accuracy.\n");
			}
		}
	}
	
	/**
	 * Function to calculate F-scores.
	 * @param beta parameter to guide preference. Use 0.5 when precision 
	 * is more important, use 2 when recall is more important.
	 */
	public static double fScore(int TP, int TN, int FP, int FN, double beta){
		double prec;
		if ((TP + FP) == 0) prec = 0;
		else prec = TP / (double)(TP + FP);
		double rec;
		if ((TP + FN) == 0) rec = 0;
		else rec = TP / (double)(TP + FN);
		double betaSq = beta*beta;
		double num = (1 + betaSq)*(prec)*(rec);
		double den = (betaSq*prec) + rec;
		if (((prec == 0) && (rec == 0)) || (den == 0))
			return -1;
		else 
			return num/den;
	}

	/**
	 * Creates a set of instances copying from a existing set.
	 * @param sourceSetOfInstances the source set of instances
	 * @param listOfInstancesIDs the list of instances ids to copy
	 */
	public static Instances createSetOfInstances(Instances sourceSetOfInstances, List<Integer> listOfInstancesIDs){
		Instances newSet = new Instances(sourceSetOfInstances);
		newSet.delete();
		int numInstances = listOfInstancesIDs.size();
		for (int i = 0; i < numInstances; i++){
			int instID = listOfInstancesIDs.get(i);
			Instance inst = sourceSetOfInstances.instance(instID);
			newSet.add(inst);
		}
		return newSet;
	}
	
	/**
	 * Creates a random training set. 
	 * @param trainingPerc the percentage of the instances used in training
	 */
	public static void createRandomTrainingAndTest(Instances trainingInstances, Instances testInstances, 
			Instances allInstances, double trainingPerc, Random r){
		//allInstances.randomize(r); 
		//Choosing training instances
		int numInstances = allInstances.numInstances();
		double percTraining = trainingPerc/100.0;
		int maxTrainingInstances = (int)(numInstances * percTraining);
		trainingInstances.delete();
		
		List<Integer> trainingList = new ArrayList<>();
		for (int i = 0; i < numInstances; i++)
			trainingList.add(i);
		Collections.shuffle(trainingList,r);
		trainingList = trainingList.subList(0, maxTrainingInstances);
		Collections.sort(trainingList);
		for (int i = maxTrainingInstances-1; i >= 0; i--){
			int instanceID = trainingList.get(i);
			Instance inst = testInstances.instance(instanceID);
			testInstances.delete(instanceID);
			trainingInstances.add(inst);
		}
	}
	
	/**
	 * Creates a random training set containing a specific percentage of positive instances. 
	 * @param trainingPerc the percentage of the instances used in training
	 * @param positivePerc the percentage of positive instances in the training set
	 */
	public static void createCustomTrainingAndTest(Instances trainingInstances, Instances testInstances, 
			Instances allInstances, double trainingPerc, double positivePerc, Random r){
		int numInstances = allInstances.numInstances();
		double percTraining = trainingPerc/100.0;
		int maxTrainingInstances = (int)(numInstances * percTraining);
		int maxPositiveInstances = (int)(maxTrainingInstances * positivePerc/100);
		int maxNegativeInstances = maxTrainingInstances - maxPositiveInstances;
		trainingInstances.delete();
		
		List<Integer> positiveIDs = new ArrayList<Integer>();
		List<Integer> negativeIDs = new ArrayList<Integer>();
		for (int instID = 0; instID < numInstances; instID++){
			Instance inst = allInstances.instance(instID);
			if (inst.classValue() == Definitions.TRUE) positiveIDs.add(instID);
			else negativeIDs.add(instID);
		}
		Collections.shuffle(positiveIDs,r);
		Collections.shuffle(negativeIDs,r);
//		if (positiveIDs.size() < maxPositiveInstances) maxPositiveInstances = positiveIDs.size();
		positiveIDs = positiveIDs.subList(0, maxPositiveInstances);
		negativeIDs = negativeIDs.subList(0, maxNegativeInstances);
			
		List<Integer> trainingList = new ArrayList<>();
		trainingList.addAll(positiveIDs);
		trainingList.addAll(negativeIDs);
		Collections.sort(trainingList);
		for (int i = maxTrainingInstances-1; i >= 0; i--){
			int instanceID = trainingList.get(i);
			Instance inst = testInstances.instance(instanceID);
			testInstances.delete(instanceID);
			trainingInstances.add(inst);
		}
	}

	/**
	 * Creates a random training set containing a specific number of positive and negative instances.
	 * @param numberOfPositives the desired number of positive instances in the training set
	 * @param numberOfNegatives the desired number of negative instances in the training set
	 */
	public static void createFixedPNTrainingAndTest(Instances trainingInstances, Instances testInstances, 
			Instances allInstances, int numberOfPositives, int numberOfNegatives, Random r){
		int numInstances = allInstances.numInstances();
		int maxTrainingInstances = numberOfPositives + numberOfNegatives;
		trainingInstances.delete();
		
		List<Integer> positiveIDs = new ArrayList<Integer>();
		List<Integer> negativeIDs = new ArrayList<Integer>();
		for (int instID = 0; instID < numInstances; instID++){
			Instance inst = allInstances.instance(instID);
			if (inst.classValue() == Definitions.TRUE) positiveIDs.add(instID);
			else negativeIDs.add(instID);
		}
		Collections.shuffle(positiveIDs,r);
		Collections.shuffle(negativeIDs,r);
		if (positiveIDs.size() < numberOfPositives) numberOfPositives = positiveIDs.size();
		if (negativeIDs.size() < numberOfNegatives) numberOfNegatives = negativeIDs.size();
		positiveIDs = positiveIDs.subList(0, numberOfPositives);
		negativeIDs = negativeIDs.subList(0, numberOfNegatives);
			
		List<Integer> trainingList = new ArrayList<>();
		trainingList.addAll(positiveIDs);
		trainingList.addAll(negativeIDs);
		Collections.sort(trainingList);
		for (int i = maxTrainingInstances-1; i >= 0; i--){
			int instanceID = trainingList.get(i);
			Instance inst = testInstances.instance(instanceID);
			testInstances.delete(instanceID);
			trainingInstances.add(inst);
		}
	}
	
	/**
	 * Return the number of positive instances in the set of instances.
	 */
	public static int countPositivesInSet(Instances setOfInstances){
		int counter = 0;
		for (int i = 0; i < setOfInstances.numInstances(); i++){
			Instance instance = setOfInstances.instance(i);
			if (instance.classValue() == Definitions.TRUE)
				counter++;
		}
		return counter;
	}

	/**
	 * Evaluate the votes of classifiers. The decision is accepted if the agreement of the votes is higher than specified.
	 */
	public static void evaluateVotes(Evaluation eval, Vote classifier, Instances setOfInstances, double minAgreement){
		Classifier[] trees = classifier.getClassifiers();
		int numClassifiers = trees.length;
		for (int instID = 0; instID < setOfInstances.numInstances(); instID++){
			Instance instance = setOfInstances.instance(instID);
			double[] votes = {0,0};
			try {
				for (int treeID = 0; treeID < numClassifiers; treeID++){
					int vote = (int)trees[treeID].classifyInstance(instance);
					votes[vote]++;
				}
				Utils.normalize(votes);
				int maxClass = Utils.maxIndex(votes);
				if (votes[maxClass] >= minAgreement){
					eval.evaluateModelOnce(maxClass, instance);
					//System.out.printf("%.3f %.3f\tPRED: %d REAL: %d\n",votes[0],votes[1],maxClass,(int)instance.classValue());
				}
			} catch (Exception e) {
				System.err.printf("Error while voting.\n");
			}
		}
	}
	
	/**
	 * Evaluate the votes of classifiers. The decision is accepted if the agreement of the votes is higher than specified.
	 */
	public static void evaluateVotes(Evaluation eval, Instances setOfInstances, double minAgreement, int[] decisionHistory, double[] agreementHistory){
		for (int instID = 0; instID < setOfInstances.numInstances(); instID++){
			try {
				if (agreementHistory[instID] >= minAgreement){
					Instance instance = setOfInstances.instance(instID);
//					if (instance.classValue() == Definitions.TRUE) System.out.printf("Positive instance\n");
					eval.evaluateModelOnce((double)decisionHistory[instID], instance);
				}
			} catch (Exception e) {
				System.err.printf("Error evaluating votes.\n");
			}
		}
	}
	
	/**
	 * Cast and tallies the votes of classifiers. All the decisions and agreement values are stored in the arrays given.
	 * @param decisionHistory the array that will store the classifiers combined decision
	 * @param agreementHistory the array that will store the agreement of decisions
	 */
	public static void castingVotes(Vote classifier, Instances setOfInstances, int[] decisionHistory, double[] agreementHistory){
		Classifier[] trees = classifier.getClassifiers();
		int numClassifiers = trees.length;
		for (int instID = 0; instID < setOfInstances.numInstances(); instID++){
			Instance instance = setOfInstances.instance(instID);
			double[] votes = {0,0};
			try {
				for (int treeID = 0; treeID < numClassifiers; treeID++){
					int vote = (int)trees[treeID].classifyInstance(instance);
					votes[vote]++;
				}
			} catch (Exception e) {
				System.err.printf("Error while casting votes.\n");
			}
			Utils.normalize(votes);
			int maxClass = Utils.maxIndex(votes);
			double maxValue = votes[maxClass];
			decisionHistory[instID] = maxClass;
			agreementHistory[instID] = maxValue;
		}
	}
	
	/**
	 * Return the indexes of elements inthe array equal or higher than the threshold given.
	 */
	public static List<Integer> getFinalDecisions(double[] agreementHistory, double minAgreement){
		int numInstances = agreementHistory.length;
		List<Integer> finals = new ArrayList<Integer>();
		for (int i = 0; i < numInstances; i++){
			if (agreementHistory[i] >= minAgreement)
				finals.add(i);
		}
		return finals;
	}
	
	/**
	 * Return the indexes of elements inthe array equal or higher than the threshold given.
	 * @param positivePredictions set this to TRUE to only get positive predictions
	 */
	public static List<Integer> getFinalDecisionsByClass(double[] agreementHistory, double minAgreement, int[] decisionHistory, boolean positivePredictions){
		int numInstances = agreementHistory.length;
		List<Integer> finals = new ArrayList<Integer>();
		int objective;
		if (positivePredictions)
			objective = Definitions.TRUE;
		else objective = Definitions.FALSE;
		for (int i = 0; i < numInstances; i++){
			if ((agreementHistory[i] >= minAgreement) && (decisionHistory[i] == objective))
				finals.add(i);
		}
		return finals;
	}
	
	/**
	 * Copy instances from the source set of instances to the target set of instances.
	 * @param instancesToCopy the IDs of instances to copy
	 * @param sourceSet source set of instances
	 * @param targetSet target set of instances
	 * @param decisionHistory the voted labels
	 */
	public static void copyInstancesWithPredictedLabel(List<Integer> instancesToCopy, Instances sourceSet, Instances targetSet, int[] decisionHistory){
		for (Integer instID: instancesToCopy){
			Instance copyInstance = (Instance) sourceSet.instance(instID).copy();
			double predictedValue = decisionHistory[instID];
			copyInstance.setClassValue(predictedValue);
			targetSet.add(copyInstance);
		}
	}
	
	/**
	 * Copy instances from the source set of instances to the target set of instances.
	 * @param instancesToCopy the IDs of instances to copy
	 * @param sourceSet source set of instances
	 * @param targetSet target set of instances
	 * @param predictedClassValue set Definitions.TRUE or Definitions.FALSE
	 */
	public static void copyInstancesWithPredictedLabel(List<Integer> instancesToCopy, Instances sourceSet, Instances targetSet, int predictedClassValue){
		for (Integer instID: instancesToCopy){
			Instance copyInstance = (Instance) sourceSet.instance(instID).copy();
			double predictedValue = (double)predictedClassValue;
			copyInstance.setClassValue(predictedValue);
			targetSet.add(copyInstance);
		}
	}
	
	/**
	 * Removes instances from a set. 
	 * @param instancesToRemove list of instances indexes that should be removed 
	 * @param setOfInstances the set of instances
	 */
	public static void removeInstancesFromSet(List<Integer> instancesToRemove, Instances setOfInstances){
		int listSize = instancesToRemove.size();
		Collections.sort(instancesToRemove);
		for (int index = listSize - 1; index >= 0; index--){
			int instID = instancesToRemove.get(index);
			setOfInstances.delete(instID);
		}
	}
	
	/**
	 * Function used to get labels from an user. The array of decisions is changed with the correct label and the array of agreement scores
	 *  receives the maximum score value. The changes are applied to instances that get correct labels from user. The indexes of these instances
	 *  are returned in a List. 
	 * @param testSet the set of instances
	 * @param userInterventionThreshold the exact agreement required to give the instance to the user
	 * @param maxDecisions the maximum number of user interventions. Set it to 0 to take unlimited user labels
	 * @return a list of instances indexes that have received labels
	 */
	public static List<Integer> userLabelSplitDecisions(Instances testSet, int[] decisionHistory, double[] agreementHistory, 
			double userInterventionThreshold, int maxDecisions){
		List<Integer> listOfUserLabels = new ArrayList<Integer>();
		int maxInstID = decisionHistory.length;
		if (maxDecisions <= 0) maxDecisions = decisionHistory.length;
		for (int instID = 0; instID < maxInstID; instID++){
			if ((agreementHistory[instID] <= userInterventionThreshold) && (decisionHistory[instID] == Definitions.TRUE)){
				int realClass = (int)testSet.instance(instID).classValue();
				decisionHistory[instID] = realClass;
				listOfUserLabels.add(instID);
				System.out.printf("positive? - class %s\n",realClass==Definitions.TRUE?"positive":"negative");
				if (listOfUserLabels.size() == maxDecisions) break;
			}
		}
		if (listOfUserLabels.size() < maxDecisions){
			for (int instID = 0; instID < maxInstID; instID++){
				if ((agreementHistory[instID] <= userInterventionThreshold) && (!listOfUserLabels.contains(instID))){
					int realClass = (int)testSet.instance(instID).classValue();
					decisionHistory[instID] = realClass;
					listOfUserLabels.add(instID);
					if (listOfUserLabels.size() == maxDecisions) break;
				}
			}
		}
		return listOfUserLabels;
	}

	/**
	 * Function that gives agreement zero to an accepted instance in the second vote but not in the first vote.
	 */
	public static void retainDecisions(List<Integer> firstRoundDecisions, List<Integer> secondRoundDecisions, double[] agreementHistory){
		for (int instID:secondRoundDecisions){
			if (!firstRoundDecisions.contains(instID))
				agreementHistory[instID] = 0;
		}
	}

	/**
	 * Returns top predictions from COMA previous execution. Predictions are sorted according to values of the <b>comaMatchersAverage</b> attribute.
	 * @param setOfInstances the set of instances
	 * @param topTrueInstances list that will contain the top instances predicted as TRUE by COMA
	 * @param maxSizeTopTrue the maximum size of the top true instances
	 * @param topFalseInstances list that will contain the top instances predicted as FALSE by COMA
	 * @param maxSizeTopFalse the maximum size of the top false instances
	 */
	public static void requestInstancesFromCOMA(Instances setOfInstances, List<Integer> topTrueInstances, int maxSizeTopTrue, 
			List<Integer> topFalseInstances, int maxSizeTopFalse){
		Attribute comaMatchersAttribute = setOfInstances.attribute("comaMatchersAverage");
		Attribute comaPredictionAttribute = setOfInstances.attribute("COMADecisionMatcher");
		List<Contender> instancesToSortPositive = new ArrayList<Contender>();
		List<Contender> instancesToSortNegative = new ArrayList<Contender>();
		int numInstances = setOfInstances.numInstances();
		for (int instID = 0; instID < numInstances; instID++){
			Instance instance = setOfInstances.instance(instID);
			if ((int)(instance.value(comaPredictionAttribute)) == Definitions.POSITIVE_COMA)
				instancesToSortPositive.add(new Contender(instID, instance.value(comaMatchersAttribute)));
			else
				instancesToSortNegative.add(new Contender(instID, instance.value(comaMatchersAttribute)));
		}
		Collections.sort(instancesToSortPositive, descendingComparator);
		if (instancesToSortPositive.size() < maxSizeTopTrue)
			maxSizeTopTrue = instancesToSortPositive.size();
		Collections.sort(instancesToSortNegative, descendingComparator);
		if (instancesToSortNegative.size() < maxSizeTopFalse)
			maxSizeTopFalse = instancesToSortNegative.size();
		for (int topK = 0; topK < maxSizeTopTrue; topK++)
			topTrueInstances.add(instancesToSortPositive.get(topK).index);
		for (int topK = 0; topK < maxSizeTopFalse; topK++)
			topFalseInstances.add(instancesToSortNegative.get(topK).index);
	}
	
	/**
	 * Returns predictions from COMA previous execution. Predictions are sorted according to values of the <b>comaMatchersAverage</b> attribute.
	 * @param setOfInstances the set of instances
	 * @param topInstances the list that will contain selected instances
	 * @param classValue set it to Definitions.TRUE or Definitions.FALSE 
	 * @param maxSize the maximum size of the list
	 * @param direction set Functions.DESCENDING or Functions.ASCENDING to sort predictions
	 */
	public static void requestInstancesFromCOMA(Instances setOfInstances, List<Integer> topInstances, int classValue, int maxSize, int direction){
		requestInstancesFromCOMA(setOfInstances, topInstances, classValue, maxSize, direction, 0);
	}
	
	public static void requestInstancesFromCOMA(Instances setOfInstances, List<Integer> topInstances, int classValue, int maxSize, int direction, int jump){
		Attribute comaMatchersAttribute = setOfInstances.attribute("comaMatchersAverage");
		Attribute comaPredictionAttribute = setOfInstances.attribute("COMADecisionMatcher");
		List<Contender> instancesToSort = new ArrayList<Contender>();
		maxSize = maxSize + jump;
		int numInstances = setOfInstances.numInstances();
		int goalClass = (classValue == Definitions.TRUE) ? Definitions.POSITIVE_COMA : Definitions.NEGATIVE_COMA;
		for (int instID = 0; instID < numInstances; instID++){
			Instance instance = setOfInstances.instance(instID);
			if ((int)(instance.value(comaPredictionAttribute)) == goalClass)
				instancesToSort.add(new Contender(instID, instance.value(comaMatchersAttribute)));
		}
		if (direction == DESCENDING)
			Collections.sort(instancesToSort, descendingComparator);
		else 
			Collections.sort(instancesToSort, ascendingComparator);
		if (instancesToSort.size() < maxSize)
			maxSize = instancesToSort.size();
		for (int topK = 0 + jump; topK < maxSize; topK++)
			topInstances.add(instancesToSort.get(topK).index);
	}
	
	/**
	 * Returns correct predictions from a Similarity Flooding previous execution. 
	 * Predictions are read from a file.
	 * @param topInstances the list that will contain the predictions
	 * @param maxSize the intended size of the list of predictions
	 * @param direction set Functions.DESCENDING or Functions.ASCENDING to sort predictions
	 * @param predictionsFilePath the predictions file path
	 * @param ic the instances controller
	 */
	public static void requestCorrectLabelInstancesFromSimFlood(List<Integer> topInstances, int maxSize, int direction, 
			String predictionsFilePath, MatchingNetworksInstancesController ic){
		List<Contender> instancesToSort = new ArrayList<Contender>();
		
		File predictionsFile = new File(predictionsFilePath);
		FileReader fr = null;
		BufferedReader br = null;
		Instances instances = ic.getPoolSet();
		try {
			fr = new FileReader(predictionsFile);
			br = new BufferedReader(fr);
			String line;
			line = br.readLine();			//reads header. ignore.	
			line = br.readLine();
			while (line != null){
				String[] parts = line.split("\t");
				int sourceID = InstancesController.getUnifiedCode(Integer.valueOf(parts[0]), Integer.valueOf(parts[1]));
				int targetID = InstancesController.getUnifiedCode(Integer.valueOf(parts[2]), Integer.valueOf(parts[3]));
				double sim = Double.valueOf(parts[4]);
				int instanceID = ic.getMatchingCandidateID(sourceID, targetID);
				Instance inst = instances.instance(instanceID);
				if (inst.classValue() == Definitions.TRUE)
					instancesToSort.add(new Contender(instanceID, sim));
				line = br.readLine();
			}
			br.close();
			fr.close();
		} catch (FileNotFoundException e) {
			System.err.printf("Error reading predictions file. [%s]\n",predictionsFilePath);
		} catch (IOException e) {
			System.err.printf("Error reading predictions file. [%s]\n",predictionsFilePath);
		}
		
		if (direction == DESCENDING)
			Collections.sort(instancesToSort, descendingComparator);
		else 
			Collections.sort(instancesToSort, ascendingComparator);
		if (instancesToSort.size() < maxSize)
			maxSize = instancesToSort.size();
		for (int topK = 0; topK < maxSize; topK++)
			topInstances.add(instancesToSort.get(topK).index);
	}
	
	public static void requestInstancesFromSimFlood(List<Integer> topInstances, int maxSize, int direction, 
			String predictionsFilePath, MatchingNetworksInstancesController ic){
		requestInstancesFromSimFlood(topInstances, maxSize, direction, predictionsFilePath, ic, 0);
	}
	
	/**
	 * Returns predictions from a Similarity Flooding previous execution. 
	 * Predictions are read from a file.
	 * @param topInstances the list that will contain the predictions
	 * @param maxSize the intended size of the list of predictions
	 * @param direction set Functions.DESCENDING or Functions.ASCENDING to sort predictions
	 * @param predictionsFilePath the predictions file path
	 * @param ic the instances controller
	 */
	public static void requestInstancesFromSimFlood(List<Integer> topInstances, int maxSize, int direction, 
			String predictionsFilePath, MatchingNetworksInstancesController ic, int jump){
		List<Contender> instancesToSort = new ArrayList<Contender>();
		maxSize = maxSize + jump;
		File predictionsFile = new File(predictionsFilePath);
		FileReader fr = null;
		BufferedReader br = null;
		try {
			fr = new FileReader(predictionsFile);
			br = new BufferedReader(fr);
			String line;
			line = br.readLine();			//reads header. ignore.	
			line = br.readLine();
			while (line != null){
				String[] parts = line.split("\t");
				int sourceID = InstancesController.getUnifiedCode(Integer.valueOf(parts[0]), Integer.valueOf(parts[1]));
				int targetID = InstancesController.getUnifiedCode(Integer.valueOf(parts[2]), Integer.valueOf(parts[3]));
				double sim = Double.valueOf(parts[4]);
				int instanceID = ic.getMatchingCandidateID(sourceID, targetID);
				instancesToSort.add(new Contender(instanceID, sim));
				line = br.readLine();
			}
			br.close();
			fr.close();
		} catch (FileNotFoundException e) {
			System.err.printf("Error reading predictions file. [%s]\n",predictionsFilePath);
		} catch (IOException e) {
			System.err.printf("Error reading predictions file. [%s]\n",predictionsFilePath);
		}
		
		if (direction == DESCENDING)
			Collections.sort(instancesToSort, descendingComparator);
		else 
			Collections.sort(instancesToSort, ascendingComparator);
		if (instancesToSort.size() < maxSize)
			maxSize = instancesToSort.size();
		for (int topK = 0 + jump; topK < maxSize; topK++)
			topInstances.add(instancesToSort.get(topK).index);
	}
	
	/**
	 * Returns predictions from COMA previous execution. 
	 * Predictions are sorted according to values of the <b>comaMatchersAverage</b> attribute.
	 * Only correct predictions are returned.
	 * @param setOfInstances the set of instances
	 * @param topInstances the list that will contain selected instances
	 * @param classValue set it to Definitions.TRUE or Definitions.FALSE 
	 * @param maxSize the maximum size of the list
	 * @param direction set Functions.DESCENDING or Functions.ASCENDING to sort predictions
	 */
	public static void requestCorrectLabelInstancesFromCOMA(Instances setOfInstances, List<Integer> topInstances, int classValue, int maxSize, int direction){
		Attribute comaMatchersAttribute = setOfInstances.attribute("comaMatchersAverage");
		Attribute comaPredictionAttribute = setOfInstances.attribute("COMADecisionMatcher");
		List<Contender> instancesToSort = new ArrayList<Contender>();
		int numInstances = setOfInstances.numInstances();
		int predictionGoalClass = (classValue == Definitions.TRUE) ? Definitions.POSITIVE_COMA : Definitions.NEGATIVE_COMA;
		int realGoalClass = classValue;
		for (int instID = 0; instID < numInstances; instID++){
			Instance instance = setOfInstances.instance(instID);
			int realClass = (int)(instance.classValue());
			int prediction = (int)(instance.value(comaPredictionAttribute));
			if ((prediction == predictionGoalClass) && (realClass == realGoalClass))
				instancesToSort.add(new Contender(instID, instance.value(comaMatchersAttribute)));
		}
		if (direction == DESCENDING)
			Collections.sort(instancesToSort, descendingComparator);
		else 
			Collections.sort(instancesToSort, ascendingComparator);
		if (instancesToSort.size() < maxSize)
			maxSize = instancesToSort.size();
		for (int topK = 0; topK < maxSize; topK++)
			topInstances.add(instancesToSort.get(topK).index);
	}
	
	/**
	 * Reads instances from a ARFF file.
	 */
	public static Instances readInstancesFromARFF(String arffFile){
		BufferedReader reader;
		Instances allInstances = null;
		try {
			reader = new BufferedReader(new FileReader(new File(arffFile)));
			allInstances = new Instances(reader);
			reader.close();
		} catch (FileNotFoundException e) {
			System.err.printf("ARFF file not found.\n");
		} catch (IOException e) {
			System.err.printf("Error reading ARFF file.\n");
		}
		return allInstances;
	}

	/**
	 * Function to measure and print statistics of a model.
	 */
	public static void measureStatistics(InstancesController ic, List<Integer> listOfPositiveIDs, String label, PrintWriter pw){
		Instances allInstances = ic.getPoolSet();
		Evaluation eval;
		try {
			eval = new Evaluation(allInstances);
			for (int instID = 0; instID < ic.getNumInstances(); instID++){
				Instance inst = allInstances.instance(instID);
				if (listOfPositiveIDs.contains(instID))
					eval.evaluateModelOnce(Definitions.TRUE, inst);
				else
					eval.evaluateModelOnce(Definitions.FALSE, inst);
			}
			double precP = eval.precision(Definitions.TRUE);
			double recP = eval.recall(Definitions.TRUE);
			double f1ScoreP = eval.fMeasure(Definitions.TRUE);
			int TP,TN,FP,FN;
			TP = (int)eval.numTruePositives(Definitions.TRUE);
			TN = (int)eval.numTrueNegatives(Definitions.TRUE); 
			FP = (int)eval.numFalsePositives(Definitions.TRUE); 
			FN = (int)eval.numFalseNegatives(Definitions.TRUE); 
			if (pw == null)
				System.out.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
			else pw.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Function to measure and print statistics of a model.
	 * It returns the IDs of instances predicted as TRUE sorted by higher certainty of the ensemble. 
	 */
	public static List<Integer> measureStatistics(Instances testInstances, Classifier classifier, String label, PrintWriter pw){
		Evaluation eval;
		List<Contender> instancesToSort = new ArrayList<Contender>();
		List<Integer> predictionsIDsSorted = new ArrayList<Integer>();
		try {
			eval = new Evaluation(testInstances);
			eval.evaluateModel(classifier, testInstances);
			//for (int instID = 0; instID < 15; instID++){
			for (int instID = 0; instID < testInstances.numInstances(); instID++){
				Instance instance = testInstances.instance(instID);
				int attributeID = MatchingNetworksInstancesController.getInstanceAttributeID(instance);
				double[] dist = classifier.distributionForInstance(instance);
				int maxIndex = Utils.maxIndex(dist);
				if (maxIndex == Definitions.TRUE)
					instancesToSort.add(new Contender(attributeID, dist[maxIndex]));
				//if ((dist[0] != 0) && (dist[0] != 1))
				//	System.out.printf("%d ->> %.3f\t%.3f\n",instID,dist[0],dist[1]);
			}
			int numPredictions = instancesToSort.size();
			Collections.sort(instancesToSort, descendingComparator);
			for (int predID = 0; predID < numPredictions; predID++){
				predictionsIDsSorted.add(instancesToSort.get(predID).index);
			}
			
			double precP = eval.precision(Definitions.TRUE);
			double recP = eval.recall(Definitions.TRUE);
			double f1ScoreP = eval.fMeasure(Definitions.TRUE);
			int TP,TN,FP,FN;
			TP = (int)eval.numTruePositives(Definitions.TRUE);
			TN = (int)eval.numTrueNegatives(Definitions.TRUE); 
			FP = (int)eval.numFalsePositives(Definitions.TRUE); 
			FN = (int)eval.numFalseNegatives(Definitions.TRUE);
			if (pw == null)
				System.out.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
			else
				pw.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return predictionsIDsSorted;
	}
	
	public static List<Integer> measureStatistics(Instances testInstances, Classifier classifier, String label, 
			List<Integer> preferentialOrdering, PrintWriter pw){
		Evaluation eval;
		List<Contender> instancesToSort = new ArrayList<Contender>();
		List<Integer> predictionsIDsSorted = new ArrayList<Integer>();
		int maxScorePref = preferentialOrdering.size();
		try {
			eval = new Evaluation(testInstances);
			eval.evaluateModel(classifier, testInstances);
			//for (int instID = 0; instID < 15; instID++){
			for (int instID = 0; instID < testInstances.numInstances(); instID++){
				Instance instance = testInstances.instance(instID);
				int attributeID = MatchingNetworksInstancesController.getInstanceAttributeID(instance);
				double[] dist = classifier.distributionForInstance(instance);
				int maxIndex = Utils.maxIndex(dist);
				if (maxIndex == Definitions.TRUE){
					int score2 = 0;
					if (preferentialOrdering.contains(instID))
						score2 = maxScorePref - preferentialOrdering.indexOf(instID);
					instancesToSort.add(new Contender(attributeID, dist[maxIndex], score2));
				}
				//if ((dist[0] != 0) && (dist[0] != 1))
				//	System.out.printf("%d ->> %.3f\t%.3f\n",instID,dist[0],dist[1]);
			}
			int numPredictions = instancesToSort.size();
			Collections.sort(instancesToSort, descendingComparator);
			for (int predID = 0; predID < numPredictions; predID++){
				predictionsIDsSorted.add(instancesToSort.get(predID).index);
			}
			
			double precP = eval.precision(Definitions.TRUE);
			double recP = eval.recall(Definitions.TRUE);
			double f1ScoreP = eval.fMeasure(Definitions.TRUE);
			int TP,TN,FP,FN;
			TP = (int)eval.numTruePositives(Definitions.TRUE);
			TN = (int)eval.numTrueNegatives(Definitions.TRUE); 
			FP = (int)eval.numFalsePositives(Definitions.TRUE); 
			FN = (int)eval.numFalseNegatives(Definitions.TRUE);
			if (pw == null)
				System.out.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
			else
				pw.printf("%s\t|\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\n",label,precP,recP,f1ScoreP,TP,TN,FP,FN);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return predictionsIDsSorted;
	}
}

