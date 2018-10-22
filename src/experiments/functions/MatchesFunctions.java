package experiments.functions;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import experiments.Definitions;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import weka.core.Instance;

public class MatchesFunctions {
	
	private static Set<Integer> acceptedMatches = new HashSet<Integer>();
	private static Set<Integer> rejectedMatches = new HashSet<Integer>();
	private static Set<Integer> inconsistentMatches = new HashSet<Integer>();
	private static Set<Integer> userConfirmedMatches = new HashSet<Integer>();
	private static Set<Integer> userRejectedMatches = new HashSet<Integer>();
	private static Set<Integer> networkRejectedMatches = new HashSet<Integer>();
	private static Set<Integer> networkAcceptedMatches = new HashSet<Integer>();
	private static int noInconsistencyCounter = 0;
	private static int inconsistencyTypeICounter = 0;
	private static int inconsistencyTypeIICounter = 0;
	
	public static void restart(){
		acceptedMatches = new HashSet<Integer>();
		rejectedMatches = new HashSet<Integer>();
		inconsistentMatches = new HashSet<Integer>();
		userConfirmedMatches = new HashSet<Integer>();
		userRejectedMatches = new HashSet<Integer>();
		networkRejectedMatches = new HashSet<Integer>();
		networkAcceptedMatches = new HashSet<Integer>();
		inconsistentMatches = new HashSet<Integer>();
		noInconsistencyCounter = 0;
		inconsistencyTypeICounter = 0;
		inconsistencyTypeIICounter = 0;
	}

	/**
	 * Inserts instances to a matches controller and returns all instances that were accepted according to network restrictions.
	 */
	public static Set<Integer> insertMatchesSequentialWithDerivingMatches(MatchesController matchesController, 
			MatchingNetworksInstancesController instancesController, List<Integer> insertInstancesIDs){
		restart();
		int numInstances = insertInstancesIDs.size();
		for (int i = 0; i < numInstances; i++){
			int instID = insertInstancesIDs.get(i);
			int code = matchesController.addMatch(instID, instancesController);
			if (code == 0){
				acceptedMatches.add(instID);
				networkAcceptedMatches.addAll(matchesController.getConstraintConfirmedMatches());
				matchesController.constraintRejectMatches(instID, instancesController);
				rejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
				networkRejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
				noInconsistencyCounter++;
			}else{
				if (matchesController.getLastConflictingMatches().size() > 0){
					inconsistentMatches.addAll(matchesController.getLastConflictingMatches());
					inconsistentMatches.add(instID);
				}
//				System.out.printf("Conflicting with %d matches.\n",matchesController.getLastConflictingMatches().size());
				networkRejectedMatches.add(instID);
				rejectedMatches.add(instID);
				if (code == 1) inconsistencyTypeICounter++;
				else inconsistencyTypeIICounter++;
			}
		}
		acceptedMatches.addAll(networkAcceptedMatches);
		rejectedMatches.addAll(networkRejectedMatches);
		return acceptedMatches;
	}

	/**
	 * Inserts instances to a matches controller if the instance does not brake any restrictions with any other candidate instances.
	 */
	public static Set<Integer> insertMatchesNoOrder(MatchesController matchesController,
			MatchingNetworksInstancesController instancesController, List<Integer> insertInstancesIDs){
		restart();
		int numInstances = insertInstancesIDs.size();
		int numSchemas = matchesController.qtSchemas();
		MatchesController mc = new MatchesController(numSchemas);
		for (int i = 0; i < numInstances; i++){
			int instID = insertInstancesIDs.get(i);
			mc.forceAddMatch(instID, instancesController, true);
		}
		List<Integer> safeInserts = new ArrayList<Integer>();
		for (int i = 0; i < numInstances; i++){
			int instID = insertInstancesIDs.get(i);
			mc.removeMatch(instID, instancesController);
			int code = mc.canAddMatch(instID, instancesController);
			if (code == 0)
				safeInserts.add(instID);
			else inconsistentMatches.add(instID);
			mc.forceAddMatch(instID, instancesController, false);
		}
		numInstances = safeInserts.size();
		for (int i = 0; i < numInstances; i++){
			int instID = safeInserts.get(i);
			int code = matchesController.addMatch(instID, instancesController);
			if (code == 0){
				acceptedMatches.add(instID);
				networkAcceptedMatches.addAll(matchesController.getConstraintConfirmedMatches());
				matchesController.constraintRejectMatches(instID, instancesController);
				networkRejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
				noInconsistencyCounter++;
			}else{
				rejectedMatches.add(instID);
				networkRejectedMatches.add(instID);
				if (code == 1) inconsistencyTypeICounter++;
				else inconsistencyTypeIICounter++;
			}
		}
		acceptedMatches.addAll(networkAcceptedMatches);
		rejectedMatches.addAll(networkRejectedMatches);
		return acceptedMatches;
	}
	
	public static Set<Integer> insertMatchesByAskingUser(MatchesController matchesController,
			MatchingNetworksInstancesController instancesController, List<Integer> uncertainInstancesIDs){
		restart();
		int numInstances = uncertainInstancesIDs.size();
		for (int i = 0; i < numInstances; i++){
			int instID = uncertainInstancesIDs.get(i);
			int code = matchesController.canAddMatch(instID, instancesController);
			if (code == -1){
				networkAcceptedMatches.add(instID);
				continue;		//already added instance, ***probably*** by a network constraint 
			}
			if (code == 0){
				Instance instance = instancesController.getPoolSet().instance(instID);
				int classValue = MatchingNetworksInstancesController.getInstanceClassValue(instance);
				if (classValue == Definitions.TRUE){
					userConfirmedMatches.add(instID);
//					System.out.printf("Confirmed by user %d\n",userConfirmedMatches.size());
					matchesController.addMatch(instID, instancesController);
//					acceptedMatches.add(instID);
					networkAcceptedMatches.addAll(matchesController.getConstraintConfirmedMatches());
					matchesController.constraintRejectMatches(instID, instancesController);
					networkRejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
					noInconsistencyCounter++;
				}else{
					userRejectedMatches.add(instID);
//					System.out.printf("....... Rejected by user %d\n",userRejectedMatches.size());
				}
			}else{
				networkRejectedMatches.add(instID);
				if (code == 1) inconsistencyTypeICounter++;
				else inconsistencyTypeIICounter++;
//				System.out.printf(".............. Rejected by constraints %d\n",networkRejectedMatches.size());
			}
		}
		acceptedMatches.addAll(userConfirmedMatches);
		acceptedMatches.addAll(networkAcceptedMatches);
		rejectedMatches.addAll(userRejectedMatches);
		rejectedMatches.addAll(networkRejectedMatches);
		return acceptedMatches;
	}
	
	/**
	 * Inserts in the network of answers all the matches that do not violate any constraint with all the other candidate instances.
	 */
	public static Set<Integer> insertConsistentMatches(MatchesController matchesController,
			MatchingNetworksInstancesController instancesController, List<Integer> matchesIDs){
		restart();
		int numInstances = matchesIDs.size();
		int numSchemas = matchesController.qtSchemas();
		MatchesController mcTemp = new MatchesController(numSchemas);
		List<Integer> safeInserts = new ArrayList<Integer>();
		
		for (int i = 0; i < numInstances; i++){
			int instID = matchesIDs.get(i);
			mcTemp.forceAddMatch(instID, instancesController, false);
		}
		for (int i = 0; i < numInstances; i++){
			int instID = matchesIDs.get(i);
			mcTemp.removeMatch(instID, instancesController);
			int code = mcTemp.canAddMatch(instID, instancesController);
			if (code > 0){
				if (code == 1) inconsistencyTypeICounter++;
				if (code == 2) inconsistencyTypeIICounter++;
				inconsistentMatches.addAll(mcTemp.getLastConflictingMatches());
				inconsistentMatches.add(instID);
			}else{
				safeInserts.add(instID);
			}
			mcTemp.forceAddMatch(instID, instancesController, false);
		}
		numInstances = safeInserts.size();
		for (int i = 0; i < numInstances; i++){
			int instID = safeInserts.get(i);
			if (inconsistentMatches.contains((Object)instID)) continue;
			int code = matchesController.addMatch(instID, instancesController);
			if (code == -1){
				networkAcceptedMatches.add(instID);
				continue;		//already added instance, ***probably*** by a network constraint
			}
			if (code == 0){
				acceptedMatches.add(instID);
				networkAcceptedMatches.addAll(matchesController.getConstraintConfirmedMatches());
				matchesController.constraintRejectMatches(instID, instancesController);
				networkRejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
				noInconsistencyCounter++;
			}else{
				System.out.printf("It should not happen.\n");
				rejectedMatches.add(instID);
				networkRejectedMatches.add(instID);
				if (code == 1) inconsistencyTypeICounter++;
				else inconsistencyTypeIICounter++;
			}
		}		
		acceptedMatches.addAll(networkAcceptedMatches);
		rejectedMatches.addAll(networkRejectedMatches);
		return acceptedMatches;
	}
	
	/**
	 * Inserts instances to a matches controller (even if it breaks a restriction) and returns all instances that
	 * were inserted.
	 */
	public static Set<Integer> forceInsertMatchesSequentialWithDerivingMatches(MatchesController matchesController, 
			MatchingNetworksInstancesController instancesController, List<Integer> insertInstancesIDs){
		restart();
		int numInstances = insertInstancesIDs.size();
		for (int i = 0; i < numInstances; i++){
			int instID = insertInstancesIDs.get(i);
			matchesController.forceAddMatch(instID, instancesController, true);
			acceptedMatches.add(instID);
			networkAcceptedMatches.addAll(matchesController.getLastForcedConstraintConfirmedMatches());
		}
		acceptedMatches.addAll(networkAcceptedMatches);
		return acceptedMatches;
	}
	
	/**
	 * Forces inconsistent instances to be added to the network. 
	 */
	public static Set<Integer> forceAddInconsistentMatches(MatchesController matchesController,
			MatchingNetworksInstancesController instancesController, List<Integer> matchesIDs){
		restart();
		int numInstances = matchesIDs.size();
		int numSchemas = matchesController.qtSchemas();
		MatchesController mcTemp = new MatchesController(numSchemas);
		List<Integer> safeInserts = new ArrayList<Integer>();
		
		for (int i = 0; i < numInstances; i++){
			int instID = matchesIDs.get(i);
			mcTemp.forceAddMatch(instID, instancesController, false);
		}
		for (int i = 0; i < numInstances; i++){
			int instID = matchesIDs.get(i);
			mcTemp.removeMatch(instID, instancesController);
			int code = mcTemp.canAddMatch(instID, instancesController);
			if (code > 0){
				if (code == 1) inconsistencyTypeICounter++;
				if (code == 2) inconsistencyTypeIICounter++;
				inconsistentMatches.addAll(mcTemp.getLastConflictingMatches());
				inconsistentMatches.add(instID);
			}else{
				safeInserts.add(instID);
			}
			mcTemp.forceAddMatch(instID, instancesController, false);
		}
		numInstances = safeInserts.size();
		for (int i = 0; i < numInstances; i++){
			int instID = safeInserts.get(i);
			if (inconsistentMatches.contains((Object)instID))
				mcTemp.forceAddMatch(instID, instancesController, false);
			int code = matchesController.addMatch(instID, instancesController);
			if (code == -1){
				networkAcceptedMatches.add(instID);
				continue;		//already added instance, ***probably*** by a network constraint
			}
			if (code == 0){
				acceptedMatches.add(instID);
				networkAcceptedMatches.addAll(matchesController.getConstraintConfirmedMatches());
				matchesController.constraintRejectMatches(instID, instancesController);
				networkRejectedMatches.addAll(matchesController.getConstraintRejectedMatches());
				noInconsistencyCounter++;
			}else{
				System.out.printf("It should not happen.\n");
				rejectedMatches.add(instID);
				networkRejectedMatches.add(instID);
				if (code == 1) inconsistencyTypeICounter++;
				else inconsistencyTypeIICounter++;
			}
		}		
		acceptedMatches.addAll(networkAcceptedMatches);
		rejectedMatches.addAll(networkRejectedMatches);
		return acceptedMatches;
	}
	
	public static Set<Integer> getInconsistentMatches(){
		return inconsistentMatches;
	}
	
	public static Set<Integer> getAcceptedMatches(){
		return acceptedMatches;
	}
	
	public static Set<Integer> getRejectedMatches(){
		return rejectedMatches;
	}
	
	public static Set<Integer> getUserConfirmedMatches(){
		return userConfirmedMatches;
	}
	
	public static Set<Integer> getUserRejectedMatches(){
		return userRejectedMatches;
	}
	
	public static Set<Integer> getNetworkRejectedMatches(){
		return networkRejectedMatches;
	}
	
	public static Set<Integer> getNetworkAcceptedMatches(){
		return networkAcceptedMatches;
	}
	
	public static int getNumberOfSuccessfulInserts(){
		return noInconsistencyCounter;
	}
	
	public static int getNumberOfInconsistenciesOfTypeI(){
		return inconsistencyTypeICounter;
	}
	
	public static int getNumberOfInconsistenciesOfTypeII(){
		return inconsistencyTypeIICounter;
	}
}

