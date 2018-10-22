package structures.schema;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MatchesController {

	private HashMap<Integer,SchemaElement> allMatches;							//<SchemaElementID, SchemaElementObject>
	private int qtSchemas;
	private ArrayList<Integer> predictedMatches,constraintConfirmedMatches,constraintRejectedMatches,forceConstraintConfirmedMatches;
	private Set<Integer> conflictingMatches;
	
	public MatchesController(int qtSchemas){
		this.qtSchemas = qtSchemas;
		allMatches = new HashMap<Integer, SchemaElement>();
	}
	
	public void restart(){
		allMatches = new HashMap<Integer, SchemaElement>();
	}
	
	/**
	 * Clones the Matches Controller containing all the matches found so far.
	 */
	public MatchesController clone(){
		MatchesController clone = new MatchesController(qtSchemas);
		Iterator<Integer> keyIterator = allMatches.keySet().iterator();
		while (keyIterator.hasNext()){
			int key = keyIterator.next();
			SchemaElement element = allMatches.get(key);
			clone.allMatches.put(key, element.clone());
		}
		return clone;
	}

	/**
	 * Returns the number of schemas.
	 */
	public int qtSchemas(){ return qtSchemas;}
	
	/**
	 * Tests a match for inconsistency of type I (one-to-one constraint). 
	 * @return Returns a list of inconsistent elements, an empty list means no inconsistencies found. 
	 */
	private ArrayList<Integer> testInconsistencyTypeI(int elementUnifiedCode1, int elementUnifiedCode2){
		SchemaElement elem1, elem2;
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		ArrayList<Integer> inconsistentElements = new ArrayList<Integer>();
		if (schema1ID == schema2ID){
			//System.out.printf("Type I conflict: Same schema #%03d\n",schema1ID);
			inconsistentElements.add(elementUnifiedCode1);
			inconsistentElements.add(elementUnifiedCode2);
		}
		
		elem1 = allMatches.get(elementUnifiedCode1);
		elem2 = allMatches.get(elementUnifiedCode2);
		
		ArrayList<Integer> inconsistencies1 = elem1.containSchema(elementUnifiedCode2);
		if (inconsistencies1!=null){
			//System.out.printf("Type I conflict: %03d[%03d] conflicting.\n",schema1ID,element1ID);
			//for (int incID: inconsistencies1) System.out.printf("\tMatched with element %03d[%03d]\n",Definitions.getSchemaID(incID),Definitions.getElementID(incID));
			inconsistentElements.add(elementUnifiedCode1);
			inconsistentElements.addAll(inconsistencies1);
		}
		ArrayList<Integer> inconsistencies2 = elem2.containSchema(elementUnifiedCode1);
		if (inconsistencies2!=null){
			//System.out.printf("Type I conflict: %03d[%03d] conflicting.\n",schema2ID,element2ID);
			//for (int incID: inconsistencies2) System.out.printf("\tMatched with element %03d[%03d]\n",Definitions.getSchemaID(incID),Definitions.getElementID(incID));
			inconsistentElements.add(elementUnifiedCode2);
			inconsistentElements.addAll(inconsistencies2);
		}
		return inconsistentElements;
	}
	
	/**
	 * Tests a match for inconsistency of type I (one-to-one constraint). 
	 * @return Returns a list of inconsistent elements, an empty list means no inconsistencies found. 
	 */
	public ArrayList<Integer> testInconsistencyTypeI(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return testInconsistencyTypeI(codes[0], codes[1]);
	}
	
	/**
	 * Tests a match for inconsistency of type II (cycle constraint).
	 * @return Returns a list of inconsistent elements, an empty list means no inconsistencies found.
	 */
	private ArrayList<Integer> testInconsistencyTypeII(int elementUnifiedCode1, int elementUnifiedCode2){
		ArrayList<ArrayList<Integer>> lists = findTail(elementUnifiedCode1);
		ArrayList<Integer> tailList1 = lists.get(0);
		ArrayList<Integer> visitedList1 = lists.get(1);
		lists = findTail(elementUnifiedCode2);
		ArrayList<Integer> tailList2 = lists.get(0);
		ArrayList<Integer> visitedList2 = lists.get(1);
		int status1;
		for (int el: tailList2)
			if (!tailList1.contains(el)) tailList1.add(el);
		for (int el: visitedList2)
			if (!visitedList1.contains(el)) visitedList1.add(el);
		//status1 = repeatedSchema(tailList1);
		status1 = repeatedSchema(visitedList1);
		if (status1 != -1){
			return tailList1;
		}
		return null;
	}
	
	/**
	 * Tests a match for inconsistency of type II (cycle constraint).
	 * @return Returns a list of inconsistent elements, an empty list means no inconsistencies found.
	 */
	public ArrayList<Integer> testInconsistencyTypeII(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return testInconsistencyTypeII(codes[0], codes[1]);
	}
	
	
	/**
	 * Adds a match, it searches for inconsistencies before committing the match.
	 * @return a code that informs what happened: <b>success</b>(0) or <b>inconsistency type I</b>(1) 
	 * or <b>inconsistency type II</b>(2).
	 * @param constraintConfirmed if this parameter is set to <b>true</b>, the match can be retrieved in the constraint confirmed matches list, 
	 * otherwise, it can be retrieved in the predicted matches list.
	 * @param deriveMatches set this parameter as <b>true</b> to derive matches with network restrictions, if possible.
	 * @return 1 or 2 if there is an inconsistency adding the match, 0 otherwise.
	 */
	private int addMatch(int elementUnifiedCode1, int elementUnifiedCode2, MatchingNetworksInstancesController ic, 
			boolean constraintConfirmed, boolean deriveMatches){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		SchemaElement elem1, elem2;
//		System.out.printf("Matching #%06d <> #%06d\n",schemaElementID1,schemaElementID2);
		if (schema1ID == schema2ID){
//			System.out.printf("Type I conflict: Same schema #%03d\n",schema1ID);
			return 1;
		}
		if (allMatches.containsKey(elementUnifiedCode1)) elem1 = allMatches.get(elementUnifiedCode1);
		else{
			elem1 = new SchemaElement(schema1ID, element1ID);
			allMatches.put(elementUnifiedCode1, elem1);
		}
		if (allMatches.containsKey(elementUnifiedCode2)) elem2 = allMatches.get(elementUnifiedCode2);
		else{
			elem2 = new SchemaElement(schema2ID, element2ID);
			allMatches.put(elementUnifiedCode2, elem2);
		}
		
		
		elem1.addMatch(elementUnifiedCode2);
		elem2.addMatch(elementUnifiedCode1);
		
		int foundInc = findInconsistencies(elementUnifiedCode1, elementUnifiedCode2, ic);
		
		if (foundInc != 0){
			elem1.removeMatch(elementUnifiedCode2);
			elem2.removeMatch(elementUnifiedCode1);
			return foundInc;
		}
		
		ArrayList<ArrayList<Integer>> lists = findTail(elementUnifiedCode1);
		ArrayList<Integer> tailList1 = lists.get(0);
		ArrayList<Integer> visitedList1 = lists.get(1);
		lists = findTail(elementUnifiedCode2);
		ArrayList<Integer> tailList2 = lists.get(0);
		ArrayList<Integer> visitedList2 = lists.get(1);

		for (int el: tailList2)
			if (!tailList1.contains(el)) tailList1.add(el);
		for (int el: visitedList2)
			if (!visitedList1.contains(el)) visitedList1.add(el);
		
		int status1 = repeatedSchema(visitedList1);
		if (status1 != -1){
			elem1.removeMatch(elementUnifiedCode2);
			elem2.removeMatch(elementUnifiedCode1);
			return 2;
		}
		
		if (!constraintConfirmed){
			int predictedID = ic.getMatchingCandidateID(elementUnifiedCode1, elementUnifiedCode2);
			predictedMatches.add(predictedID);
		}else{
			int matchConfirmedID = ic.getMatchingCandidateID(elementUnifiedCode1, elementUnifiedCode2);
			constraintConfirmedMatches.add(matchConfirmedID);
		}
		
		//recommendations
		if (deriveMatches){
			for (int schElID: visitedList1){
				if (schElID != elementUnifiedCode1)
					if (!elem1.isMatched(schElID))
						//System.out.printf("Recommended match: #%06d <> #%06d\n",schElID,schemaElementID1);
						addMatch(elementUnifiedCode1, schElID, ic, true, true);
					
				if (schElID != elementUnifiedCode2)
					if (!elem2.isMatched(schElID))
						//System.out.printf("Recommended match: #%06d <> #%06d\n",schElID,schemaElementID2);
						addMatch(elementUnifiedCode2, schElID, ic, true, true);
			}
		}
		return 0;
	}
	
	@Deprecated
	private int addMatchBackUp(int elementUnifiedCode1, int elementUnifiedCode2, MatchingNetworksInstancesController ic, boolean constraintConfirmed){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		SchemaElement elem1, elem2;
//		System.out.printf("Matching #%06d <> #%06d\n",schemaElementID1,schemaElementID2);
		if (schema1ID == schema2ID){
//			System.out.printf("Type I conflict: Same schema #%03d\n",schema1ID);
			return 1;
		}
		if (allMatches.containsKey(elementUnifiedCode1)) elem1 = allMatches.get(elementUnifiedCode1);
		else{
			elem1 = new SchemaElement(schema1ID, element1ID);
			allMatches.put(elementUnifiedCode1, elem1);
		}
		if (allMatches.containsKey(elementUnifiedCode2)) elem2 = allMatches.get(elementUnifiedCode2);
		else{
			elem2 = new SchemaElement(schema2ID, element2ID);
			allMatches.put(elementUnifiedCode2, elem2);
		}
		
		ArrayList<Integer> inconsistencies = elem1.containSchema(elementUnifiedCode2);
		if (inconsistencies!=null){
//			System.out.printf("Type I conflict: %d[%d] conflicting.\n",element1ID,schema1ID);
//			for (int incID: inconsistencies) System.out.printf("\tMatched with element %d[%d]\n",Definitions.getElementID(incID),Definitions.getSchemaID(incID));
			
			return 1;
		}
		inconsistencies = elem2.containSchema(elementUnifiedCode1);
		if (inconsistencies!=null){
//			System.out.printf("Type I conflict: %d[%d] conflicting.\n",element2ID,schema2ID);
//			for (int incID: inconsistencies) System.out.printf("\tMatched with element %d[%d]\n",Definitions.getElementID(incID),Definitions.getSchemaID(incID));
			return 1;
		}
		elem1.addMatch(elementUnifiedCode2);
		elem2.addMatch(elementUnifiedCode1);
		
		//System.out.printf("Added match between %06d and %06d ",elementUnifiedCode1,elementUnifiedCode2);
		//if (constraintConfirmed) System.out.printf("Constraint-based match.");
		//System.out.printf("\n");
		
		ArrayList<ArrayList<Integer>> lists = findTail(elementUnifiedCode1);
		ArrayList<Integer> tailList1 = lists.get(0);
		ArrayList<Integer> visitedList1 = lists.get(1);
		lists = findTail(elementUnifiedCode2);
		ArrayList<Integer> tailList2 = lists.get(0);
		ArrayList<Integer> visitedList2 = lists.get(1);

		for (int el: tailList2)
			if (!tailList1.contains(el)) tailList1.add(el);
		for (int el: visitedList2)
			if (!visitedList1.contains(el)) visitedList1.add(el);
		
//		for (int el:tailList1) System.out.printf("TL1: %06d\n",el);
//		for (int el:tailList2) System.out.printf("TL2: %06d\n",el);
//		System.out.printf("\n");
		
//		status1 = repeatedSchema(tailList1);
		int status1 = repeatedSchema(visitedList1);
		if (status1 != -1){
//			System.out.printf("Type II conflict: conflicting schema #%03d\n",status1);
//			System.out.printf("Type II conflict: conflicting schema #%03d\t",status1);
//			InconsistenciesNetworkMatching.printSchema(status1);
//			System.out.printf("\n");
			elem1.removeMatch(elementUnifiedCode2);
			elem2.removeMatch(elementUnifiedCode1);
			return 2;
		}
//		status1 = repeatedSchema(tailList2);
//		if (status1 != -1){
//			System.out.printf("Type II conflict: conflicting schema #%03d\n",status1);
//			elem1.removeMatch(schemaElementID2);
//			elem2.removeMatch(schemaElementID1);
//			return 2;
//		}
		if (!constraintConfirmed){
			int predictedID = ic.getMatchingCandidateID(elementUnifiedCode1, elementUnifiedCode2);
			predictedMatches.add(predictedID);
		}else{
			int matchConfirmedID = ic.getMatchingCandidateID(elementUnifiedCode1, elementUnifiedCode2);
			constraintConfirmedMatches.add(matchConfirmedID);
//			System.out.printf("Deriving Match: %d[sch-%d] -- %d[sch-%d]\n",);
		}
		//recommendations
		for (int schElID: visitedList1){
			if (schElID != elementUnifiedCode1)
				if (!elem1.isMatched(schElID))
					//System.out.printf("Recommended match: #%06d <> #%06d\n",schElID,schemaElementID1);
					addMatch(elementUnifiedCode1, schElID, ic, true, true);
				
			if (schElID != elementUnifiedCode2)
				if (!elem2.isMatched(schElID))
					//System.out.printf("Recommended match: #%06d <> #%06d\n",schElID,schemaElementID2);
					addMatch(elementUnifiedCode2, schElID, ic, true, true);
		}
//		blockMatches(schemaElementID1,schemaElementID2);
		return 0;
	}

	/**
	 * Adds a match, it searches for inconsistencies before committing the match. <br>
	 * By default, derived matches are generated.
	 * @return a code that informs what happened: <b>success</b>(0) or <b>inconsistency type I</b>(1) 
	 * or <b>inconsistency type II</b>(2).
	 */
	public int addMatch(int instanceID, MatchingNetworksInstancesController ic){
		return addMatch(instanceID, ic, true);
	}
	
	/**
	 * Adds a match, it searches for inconsistencies before committing the match.
	 * @param deriveMatches set this parameter to <b>true</b> to generate derived matches, if possible.
	 * @return a code that informs what happened: <b>success</b>(0) or <b>inconsistency type I</b>(1) 
	 * or <b>inconsistency type II</b>(2).
	 */
	public int addMatch(int instanceID, MatchingNetworksInstancesController ic, boolean deriveMatches){
		predictedMatches = new ArrayList<Integer>();
		constraintConfirmedMatches = new ArrayList<Integer>();
		conflictingMatches = new HashSet<Integer>();
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return addMatch(codes[0],codes[1], ic, false, deriveMatches);
	}
	
//	@Deprecated
//	/**
//	 * Adds a match, it searches for inconsistencies before committing the match.
//	 * @return a code that informs what happened: <b>success</b>(0) or <b>inconsistency type I</b>(1) 
//	 * or <b>inconsistency type II</b>(2).
//	 * matchCandidate mask: ElementID[SchemaID]-ElementID[SchemaID]
//	 */
//	public int addMatch(String matchingCandidate){
//		predictedMatches = new ArrayList<Integer>();
//		constraintConfirmedMatches = new ArrayList<Integer>();
//		int[] codes = MatchingNetworksInstancesController.getInstanceElementsCodes(matchingCandidate);
//		//System.out.printf("Force-adding Match: %d[sch-%d] -- %d[sch-%d]\n",element1ID,schema1ID,element2ID,schema2ID);
//		return addMatch(codes[0],codes[1], false);
//	}
	
	/**
	 * It searches for all the matches that brake any of the constraints and mark them as rejected.
	 */
	public ArrayList<Integer> constraintRejectMatches(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return constraintRejectMatches(codes[0],codes[1], ic);
	}
	
	/**
	 * It searches for all the matches that brake any of the constraints and mark them as rejected.
	 */
	private ArrayList<Integer> constraintRejectMatches(int elementUnifiedCode1, int elementUnifiedCode2, MatchingNetworksInstancesController ic){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		
		constraintRejectedMatches = new ArrayList<Integer>();
		int maxElementsSchema1, maxElementsSchema2;
		maxElementsSchema1 = ic.getQuantityOfElementsFromSchema(schema1ID);
		maxElementsSchema2 = ic.getQuantityOfElementsFromSchema(schema2ID);
		
		for (int targetID=0; targetID<=maxElementsSchema2; targetID++){
			if (targetID == element2ID) continue;
			int targetUnifiedCode = MatchingNetworksInstancesController.getUnifiedCode(schema2ID, targetID);
//			int blockMatchID = ic.getMatchingCandidateID(schema1ID, element1ID, schema2ID, targetID);
			int blockMatchID = ic.getMatchingCandidateID(elementUnifiedCode1, targetUnifiedCode);
			if (blockMatchID != -1)
				constraintRejectedMatches.add(blockMatchID);
//				System.out.printf("*********Blocked match between %06d and %06d.\n",elementUnifiedCode1, targetUnifiedCode);
		}
		for (int sourceID=0; sourceID<=maxElementsSchema1; sourceID++){
			if (sourceID == element1ID) continue;
			int sourceUnifiedCode = MatchingNetworksInstancesController.getUnifiedCode(schema1ID, sourceID);
//			int blockMatchID = ic.getMatchingCandidateID(schema1ID, sourceID, schema2ID, element2ID);
			int blockMatchID = ic.getMatchingCandidateID(sourceUnifiedCode, elementUnifiedCode2);
			if (blockMatchID != -1)
				constraintRejectedMatches.add(blockMatchID);
//				System.out.printf("Blocked match between %06d and %06d.\n",sourceUnifiedCode, elementUnifiedCode2);
		}
		return constraintRejectedMatches;
	}
	
//	@Deprecated
//	/**
//	 * It searches for all the matches that brake any of the constraints and mark them as rejected.
//	 */
//	public ArrayList<Integer> constraintRejectMatches(String matchingCandidate, MatchingNetworksInstancesController ic){
//		int[] codes = ic.getInstanceUnifiedCodes(matchingCandidate);
//		return constraintRejectMatches(codes[0],codes[1], ic);
//	}
	
	/**
	 * Finds all the tails (or leafs) from a specified element<br>
	 * Returns two lists: a list of all elements visited and the list of the farthest (deepest) elements.
	 */
	private ArrayList<ArrayList<Integer>> findTail(int schemaElementID){
		SchemaElement element;
		if (allMatches.containsKey(schemaElementID)) element = allMatches.get(schemaElementID);
		else return null;
		int schemaID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(schemaElementID);
		ArrayList<Integer> list = new ArrayList<Integer>();
		ArrayList<Integer> visited = new ArrayList<Integer>();
		list.add(schemaElementID);
		for (int i=0; i<list.size(); i++){
			schemaElementID = list.get(i);
			if (visited.contains(schemaElementID)){
				list.remove(i);
				i--;
				continue;
			}
			element = allMatches.get(schemaElementID);
			visited.add(schemaElementID);
			if (element.qtMatches() > 1){			//not tail
				list.remove(i);
				i--;
			}
			list.addAll(element.getMatches());
		}
		ArrayList<ArrayList<Integer>> lists = new ArrayList<ArrayList<Integer>>();
		lists.add(list);
		lists.add(visited);
		return lists;
	}
	
	/**
	 * It finds any inconsistencies of types I or II caused by the insertion of<br>
	 *  the specified matching between source and target elements.
	 * It returns the code of the inconsistency, if existent, or zero.
	 */
	public int findInconsistencies(int schemaElementIDSource, int schemaElementIDTarget, 
			MatchingNetworksInstancesController ic){
		List<List<Integer>> foundPaths = new ArrayList<List<Integer>>();
		int targetSchemaID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(schemaElementIDSource);
		List<Integer> currentPath = new ArrayList<Integer>();
		List<Integer> visited = new ArrayList<Integer>();
		findingPaths(schemaElementIDSource, currentPath, visited, targetSchemaID, foundPaths);
		//System.out.printf("visited: %s\n",visited.toString());
		//System.out.printf("~~~~~~~~~~~~~~~~~~~~~\n");
		boolean fourPointsPath = false;
		for (List<Integer> path: foundPaths){
			//System.out.printf("Path inconsistente: %s\n",path.toString());
			int match = ic.getMatchingCandidateID(path.get(0),path.get(1));
			conflictingMatches.add(match);
			//System.out.printf("added match inc: %d\n",match);
			int size = path.size();
			match = ic.getMatchingCandidateID(path.get(size-2),path.get(size-1));
			conflictingMatches.add(match);
			if (size > 3) fourPointsPath = true;
			//System.out.printf("added match inc: %d\n",match);
			//System.out.printf("----------------------------------------------\n");
		}
		
		
		targetSchemaID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(schemaElementIDTarget);
		currentPath = new ArrayList<Integer>();
		visited = new ArrayList<Integer>();
		findingPaths(schemaElementIDTarget, currentPath, visited, targetSchemaID, foundPaths);
		//System.out.printf("visited: %s\n",visited.toString());
		//fourPointsPath = false;		definido como false mais acima
		for (List<Integer> path: foundPaths){
			//System.out.printf("Path inconsistente: %s\n",path.toString());
			int match = ic.getMatchingCandidateID(path.get(0),path.get(1));
			conflictingMatches.add(match);
			//System.out.printf("added match inc: %d\n",match);
			int size = path.size();
			match = ic.getMatchingCandidateID(path.get(size-2),path.get(size-1));
			conflictingMatches.add(match);
			if (size > 3) fourPointsPath = true;
			//System.out.printf("added match inc: %d\n",match);
			//System.out.printf("----------------------------------------------\n");
		}
		if (foundPaths.size() == 0) return 0;
		else{
			int match = ic.getMatchingCandidateID(schemaElementIDSource,schemaElementIDTarget);
			conflictingMatches.remove((Object)match);
			if (fourPointsPath)
				return 2;
			else return 1;
		}
	}

	//***TESTING FUNCTION
	public int findInconsistencies2(int schemaElementIDSource, int schemaElementIDTarget, 
			MatchingNetworksInstancesController ic){
		List<Integer> visitedElements = new ArrayList<Integer>();
		Set<Integer> deadEnds = new HashSet<Integer>();
		findDeadEnds(schemaElementIDSource, visitedElements, deadEnds);
		System.out.printf("Dead ends: %s\n",deadEnds.toString());
//		a partir do dead end, tracar todos os caminhos, busca profundidade.
		return 0;
	}
	
	//***TESTING FUNCTION
	/**
	 * Function is used to find paths from a starting point to any point at the target schema. 
	 * @param startElementID the id of the starting point
	 * @param currentPath it saves the current path, it should start blank
	 * @param visitedElements it saves visited elements, it should start blank
	 * @param targetSchemaID the id of the target schema
	 * @param foundPaths it saves all paths found, it should start blank
	 */
	private void findingPaths2(int startElementID, List<Integer> currentPath, List<Integer> visitedElements,
			int targetSchemaID, List<List<Integer>> foundPaths){
		if (allMatches.containsKey(startElementID)){ 
			SchemaElement element;
			visitedElements.add(startElementID);
			//System.out.printf("------------------------\n");
			//System.out.printf("Visiting: %06d\n",startElementID);
			currentPath.add(startElementID);
			System.out.printf("CurrentPath: %s\n",currentPath.toString());
			element = allMatches.get(startElementID);
			List<Integer> children = element.getMatches();
			for (int child:children){
				if (!visitedElements.contains(child)){
					int schemaID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(child);
					if (schemaID == targetSchemaID){
						List<Integer> newPath = new ArrayList<Integer>();
						newPath.addAll(currentPath);
						newPath.add(child);
						foundPaths.add(newPath);
					}else{
						findingPaths(child, currentPath, visitedElements, targetSchemaID, foundPaths);
					}
				}
			}
			currentPath.remove((Object)startElementID);
		}
	}
	
	//***TESTING FUNCTION
	private void findDeadEnds(int currentElementID, List<Integer> visitedElements, Set<Integer> deadEnds){
		if (allMatches.containsKey(currentElementID)){
			SchemaElement element;
			visitedElements.add(currentElementID);
			element = allMatches.get(currentElementID);
			List<Integer> children = element.getMatches();
			children.removeAll(visitedElements);
			if (children.size() == 0)
				deadEnds.add(currentElementID);
			else{
				for (int child:children){
					findDeadEnds(child, visitedElements, deadEnds);
				}
			}
		}
	}
	
	/**
	 * Function is used to find paths from a starting point to any point at the target schema. 
	 * @param startElementID the id of the starting point
	 * @param currentPath it saves the current path, it should start blank
	 * @param visitedElements it saves visited elements, it should start blank
	 * @param targetSchemaID the id of the target schema
	 * @param foundPaths it saves all paths found, it should start blank
	 */
	private void findingPaths(int startElementID, List<Integer> currentPath, List<Integer> visitedElements,
			int targetSchemaID, List<List<Integer>> foundPaths){
		if (allMatches.containsKey(startElementID)){ 
			SchemaElement element;
			visitedElements.add(startElementID);
			//System.out.printf("------------------------\n");
			//System.out.printf("Visiting: %06d\n",startElementID);
			currentPath.add(startElementID);
			//System.out.printf("CurrentPath: %s\n",currentPath.toString());
			element = allMatches.get(startElementID);
			List<Integer> children = element.getMatches();
			for (int child:children){
				if (!visitedElements.contains(child)){
					int schemaID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(child);
					if (schemaID == targetSchemaID){
						List<Integer> newPath = new ArrayList<Integer>();
						newPath.addAll(currentPath);
						newPath.add(child);
						foundPaths.add(newPath);
					}else{
						findingPaths(child, currentPath, visitedElements, targetSchemaID, foundPaths);
					}
				}
			}
			currentPath.remove((Object)startElementID);
		}
	}
	
	/**
	 * Verifies if a schema is repeated among a list of elements.
	 * @return -1 if no schema is repeated or the first repeated schema ID found.  
	 */
	private int repeatedSchema(ArrayList<Integer> list){
		ArrayList<Integer> schemasID = new ArrayList<Integer>();
		for (int elemID: list){
			int schemaID = InstancesController.getSchemaIDFromUnifiedCode(elemID);
			if (schemasID.contains(schemaID)) return schemaID;
			else schemasID.add(schemaID);
		}
		return -1;
	}
	
	/**
	 * Adds a match without searching for inconsistencies.
	 */
	private void forceAddMatch(int elementUnifiedCode1, int elementUnifiedCode2, MatchingNetworksInstancesController ic, 
			boolean constraintConfirmed, boolean deriveMatches){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		SchemaElement elem1, elem2;
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		
		if (schema1ID == schema2ID){
			return;
		}
		
		
		if (allMatches.containsKey(elementUnifiedCode1)) elem1 = allMatches.get(elementUnifiedCode1);
		else{
			elem1 = new SchemaElement(schema1ID, element1ID);
			allMatches.put(elementUnifiedCode1, elem1);
		}
		
		if (!elem1.isMatched(elementUnifiedCode2)){
		
			if (allMatches.containsKey(elementUnifiedCode2)) elem2 = allMatches.get(elementUnifiedCode2);
			else{
				elem2 = new SchemaElement(schema2ID, element2ID);
				allMatches.put(elementUnifiedCode2, elem2);
			}
			elem1.addMatch(elementUnifiedCode2);
			elem2.addMatch(elementUnifiedCode1);
			
			ArrayList<ArrayList<Integer>> lists = findTail(elementUnifiedCode1);
			ArrayList<Integer> tailList1 = lists.get(0);
			ArrayList<Integer> visitedList1 = lists.get(1);
			lists = findTail(elementUnifiedCode2);
			ArrayList<Integer> tailList2 = lists.get(0);
			ArrayList<Integer> visitedList2 = lists.get(1);
	
			for (int el: tailList2)
				if (!tailList1.contains(el)) tailList1.add(el);
			for (int el: visitedList2)
				if (!visitedList1.contains(el)) visitedList1.add(el);
			
			int status1 = repeatedSchema(visitedList1);
			//recommendations
			
			if (constraintConfirmed){
				int predictedID = ic.getMatchingCandidateID(elementUnifiedCode1, elementUnifiedCode2);
				forceConstraintConfirmedMatches.add(predictedID);
			}
			
			if (deriveMatches){
				for (int schElID: visitedList1){
					if (schElID != elementUnifiedCode1)
						if (!elem1.isMatched(schElID))
							forceAddMatch(elementUnifiedCode1, schElID, ic, true, deriveMatches);
						
					if (schElID != elementUnifiedCode2)
						if (!elem2.isMatched(schElID))
							forceAddMatch(elementUnifiedCode2, schElID, ic, true, deriveMatches);
				}
			}
		}
	}
	
	/**
	 * Adds a match without searching for inconsistencies.
	 */
	public void forceAddMatch(int instanceID, MatchingNetworksInstancesController ic, boolean deriveMatches){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		forceConstraintConfirmedMatches = new ArrayList<Integer>();
		if (!isMatch(instanceID, ic))
			forceAddMatch(codes[0],codes[1],ic,false,deriveMatches);
	}

	/**
	 * Verifies if a match can be added without generating any inconsistencies.
	 * @return a code that informs what would happen: <b>it can be added</b>(0) or <br>
	 * <b>will generate an inconsistency of type I</b>(1) or <br> 
	 * <b>will generate an inconsistency of type II</b>(2).
	 */
	public int canAddMatch(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		conflictingMatches = new HashSet<Integer>();
		if (isMatch(instanceID, ic))
			return -1;
		return tryAddMatch(codes[0],codes[1],ic);
	}
	
	/**
	 * Removes a match, if exists.
	 * @return -1 if the match is impossible (same schema), 0 if no match is removed, 1 if a match is removed
	 */
	public int removeMatch(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return removeMatch(codes[0],codes[1]);
	}
	
	/**
	 * Removes a match, if exists.
	 * @return -1 if the match is impossible (same schema), 0 if no match is removed, 1 if a match is removed
	 */
	private int removeMatch(int elementUnifiedCode1, int elementUnifiedCode2){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		SchemaElement elem1, elem2;
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		
		if (schema1ID == schema2ID) return -1;
		
		if (allMatches.containsKey(elementUnifiedCode1)) elem1 = allMatches.get(elementUnifiedCode1);
		else
			return 0;
		if (allMatches.containsKey(elementUnifiedCode2)) elem2 = allMatches.get(elementUnifiedCode2);
		else
			return 0;
		
		elem1.removeMatch(elementUnifiedCode2);
		elem2.removeMatch(elementUnifiedCode1);
		return 1;
	}
	
	/**
	 * Function that simulates adding the match to verify if it would cause an inconsistency.
	 * @return the code of the inconsistency, if it would happen. 0 otherwise.
	 */
	private int tryAddMatch(int elementUnifiedCode1, int elementUnifiedCode2,MatchingNetworksInstancesController ic){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		SchemaElement elem1, elem2;
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		
		if (allMatches.containsKey(elementUnifiedCode1)) elem1 = allMatches.get(elementUnifiedCode1);
		else{
			elem1 = new SchemaElement(schema1ID, element1ID);
			allMatches.put(elementUnifiedCode1, elem1);
		}
		if (allMatches.containsKey(elementUnifiedCode2)) elem2 = allMatches.get(elementUnifiedCode2);
		else{
			elem2 = new SchemaElement(schema2ID, element2ID);
			allMatches.put(elementUnifiedCode2, elem2);
		}
		
		elem1.addMatch(elementUnifiedCode2);
		elem2.addMatch(elementUnifiedCode1);
		
		int foundInc = findInconsistencies(elementUnifiedCode1, elementUnifiedCode2, ic);
		
		elem1.removeMatch(elementUnifiedCode2);
		elem2.removeMatch(elementUnifiedCode1);
			
		return foundInc;
	}

	/**
	 * The last conflicting matches list contains the IDs of conflicting matching elements
	 *  from the last match operation in the controller. <br>
	 */
	public Set<Integer> getLastConflictingMatches(){
		return conflictingMatches;
	}

	/**
	 * The last predicted matches list contains the IDs of predicted matching elements
	 *  from the last match operation in the controller. <br>
	 */
	public List<Integer> getLastPredictedMatches(){
		return predictedMatches;
	}
	
	/**
	 * The forced constraint confirmed matches list contains the IDs of constraint-confirmed matching elements
	 *  from the last forced-add match operation in the controller. <br>
	 */
	public List<Integer> getLastForcedConstraintConfirmedMatches(){
		return forceConstraintConfirmedMatches;
	}
	
	/**
	 * The constraint confirmed matches list contains the IDs of constraint-confirmed matching elements
	 *  from the last match operation in the controller. <br>
	 */
	public List<Integer> getConstraintConfirmedMatches(){
		return constraintConfirmedMatches;
	}
	
	/**
	 * The constraint rejected matches list contains the IDs of constraint-rejected matching elements
	 *  from the last match operation in the controller. <br>
	 */
	public List<Integer> getConstraintRejectedMatches(){
		return constraintRejectedMatches;
	}
	
	/**
	 * Verifies if the matching between elements is present in the controller.
	 */
	private boolean isMatch(int elementUnifiedCode1, int elementUnifiedCode2){
		if (!allMatches.containsKey(elementUnifiedCode1)) return false;
		SchemaElement element = allMatches.get(elementUnifiedCode1);
		return element.isMatched(elementUnifiedCode2);
	}
	
	/**
	 * Verifies if the matching between elements is present in the controller.
	 */
	public boolean isMatch(int instanceID, MatchingNetworksInstancesController ic){
		int[] codes = ic.getInstanceUnifiedCodes(instanceID);
		return isMatch(codes[0],codes[1]);
	}
	
	private int[] countersMatchesNumberBySchema;

	/**
	 * Generates an array containing the number of elements are matched in each schema.
	 * It considers the matches already added in the matches controller.
	 */
	public int[] countMatchesBySchema(InstancesController instancesController){
		countersMatchesNumberBySchema = new int[qtSchemas];
		for (int i = 0; i < qtSchemas; i++) countersMatchesNumberBySchema[i] = 0;
		Set<Integer> keys = allMatches.keySet();
		for (int schemaElementCodeSource : keys){
			SchemaElement match = allMatches.get(schemaElementCodeSource);
			int schemaIDsource = match.getSchemaID();
			ArrayList<Integer> matches = match.getMatches();
			for (int schemaElementCodeTarget : matches){
				int schemaIDtarget = InstancesController.getSchemaIDFromUnifiedCode(schemaElementCodeTarget);
				countersMatchesNumberBySchema[schemaIDsource]++;
				countersMatchesNumberBySchema[schemaIDtarget]++;
			}
		}
		for (int i = 0; i < qtSchemas; i++) countersMatchesNumberBySchema[i] = countersMatchesNumberBySchema[i]/2;
		return countersMatchesNumberBySchema;
	}
	
	/**
	 * Generates an array containing the number of elements are matched in each schema. 
	 * Only the matches in the array are considered. 
	 */
	public int[] countMatchesBySchema(int numberOfSchemas, ArrayList<Integer> predictedMatchesIDs, 
			MatchingNetworksInstancesController instancesController){
		if (predictedMatchesIDs.contains(-1)) predictedMatchesIDs.remove((Object)(-1));
		countersMatchesNumberBySchema = new int[numberOfSchemas];	
		for (int i=0; i<countersMatchesNumberBySchema.length; i++) countersMatchesNumberBySchema[i] = 0;
		for (int instID: predictedMatchesIDs){
			int[] schElemIDs = instancesController.getInstanceUnifiedCodes(instID);
			int schemaID = InstancesController.getSchemaIDFromUnifiedCode(schElemIDs[0]);
			countersMatchesNumberBySchema[schemaID]++;
			schemaID = InstancesController.getSchemaIDFromUnifiedCode(schElemIDs[1]);
			countersMatchesNumberBySchema[schemaID]++;
		}
		return getMatchesBySchemaCounters();
	}
	
	/**
	 * Get the number of elements are matched in each schema. It should be called after 
	 * {@link MatchesController#countMatchesBySchema(InstancesController)} or {@link MatchesController#countMatchesBySchema(int, ArrayList, MatchingNetworksInstancesController)}
	 */
	public int[] getMatchesBySchemaCounters(){ return countersMatchesNumberBySchema;}
	
	/**
	 * It retrieves the schemas with less elements matched.
	 * @param perc Maximum percentage of elements matched accepted by schema.
	 */
	public ArrayList<Integer> getLessMatchedSchema(double perc){
		ArrayList<Integer> schemasIDs = new ArrayList<Integer>();
		int sumOfElementsMatched = 0;
		int lowerValue = 9999999;
		for (int i=0; i<countersMatchesNumberBySchema.length; i++) sumOfElementsMatched += countersMatchesNumberBySchema[i];
		for (int i=0; i<countersMatchesNumberBySchema.length; i++){
			if (countersMatchesNumberBySchema[i] < perc*sumOfElementsMatched)
				schemasIDs.add(i);
			if (countersMatchesNumberBySchema[i] < lowerValue)
				lowerValue = countersMatchesNumberBySchema[i];
		}
		if (schemasIDs.size()==0)
			for (int i=0; i<countersMatchesNumberBySchema.length; i++)
				if (countersMatchesNumberBySchema[i] == lowerValue)
					schemasIDs.add(i);
		return schemasIDs;
	}
	
	public class ValueComparator implements Comparator<Integer> {
	    Map<Integer, Integer> base;
	    public ValueComparator(Map<Integer, Integer> base) {
	        this.base = base;
	    }
	    // Note: this comparator imposes orderings that are inconsistent with equals.    
	    public int compare(Integer a, Integer b) {
	        if (base.get(a) >= base.get(b)) {
	            return -1;
	        } else {
	            return 1;
	        } // returning 0 would merge keys
	    }
	}
	
}
