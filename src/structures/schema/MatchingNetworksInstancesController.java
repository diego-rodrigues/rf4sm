package structures.schema;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class MatchingNetworksInstancesController extends InstancesController{
	
	public static final int CONSTRAINT_CONFIRMED = 4;
	public static final int CONSTRAINT_REJECTED = 5;
	
	private Instances constraintRejectedSet;
	private Instances constraintConfirmedSet;
	
	private List<Integer> constraintRejectedIDs;						//instances blocked by structural inconsistencies
	private List<Integer> constraintConfirmedIDs;						//instances added to training by structural derivation

	public MatchingNetworksInstancesController(String arffFilePath, int qtSchemas){
		super(arffFilePath,qtSchemas);
		constraintRejectedIDs = new ArrayList<Integer>();
		constraintConfirmedIDs = new ArrayList<Integer>();
	}
	
	/**
	 * Returns the hash codes from the matching candidate elements. <br>
	 * Array: SSA | EEA | SSB | EEB <br>
	 * It returns codes in the form SSSEEE. <b>SSS</b> is the schema hash code and <b>EEE</b> is the element hash code. 
	 */
	protected int[] getInstanceElementsCodes(int instanceID){
		String candidate = getMatchingCandidateLabel(instanceID);
		int[] parts = new int[4];
		String candpt1 = candidate.split("-")[0];
		String candpt2 = candidate.split("-")[1];
		parts[0] = Integer.parseInt(candpt1.split("\\[")[1].split("\\]")[0]);		//source schema code 
		parts[1] = Integer.parseInt(candpt1.split("\\[")[0]); 						//source element code
		parts[2] = Integer.parseInt(candpt2.split("\\[")[1].split("\\]")[0]);		//target schema code
		parts[3] = Integer.parseInt(candpt2.split("\\[")[0]);						//target element code
		return parts;
		// EEE[SSS]-EEE[SSS]   : code format
	}
	
	/**
	 * Returns the universal (schema+element) hash codes from the matching candidate elements. <br>
	 * It returns an array of codes in the form SSAEEA | SSBEEB.<br> 
	 * <b>SSAEEA</b> is the source hash code (schema SSA and element EEA). <br>
	 * <b>SSBEEB</b> is the target hash code (schema SSB and element EEB). <br>
	 */
	public int[] getInstanceUnifiedCodes(int instanceID){
		String candidate = getMatchingCandidateLabel(instanceID);
		if (candidate == null) return null;
		int[] parts = new int[2];
		String candpt1 = candidate.split("-")[0];
		String candpt2 = candidate.split("-")[1];
		parts[0] = Integer.parseInt(candpt1.split("\\[")[0]) + Integer.parseInt(candpt1.split("\\[")[1].split("\\]")[0])*1000;
		parts[1] = Integer.parseInt(candpt2.split("\\[")[0]) + Integer.parseInt(candpt2.split("\\[")[1].split("\\]")[0])*1000;
		return parts;
	}
	
//	@Deprecated
//	/**
//	 * Returns the universal (schema+element) hash codes from the matching candidate elements. <br>
//	 * It returns an array of codes in the form SSAEEA | SSBEEB.<br> 
//	 * <b>SSAEEA</b> is the source hash code (schema SSA and element EEA). <br>
//	 * <b>SSBEEB</b> is the target hash code (schema SSB and element EEB). <br>
//	 */
//	public int[] getInstanceUnifiedCodes(String matchingCandidateString){
//		int instID = getMatchingCandidateID(matchingCandidateString);
//		return getInstanceUnifiedCodes(instID);
//	}
	
	
	//atencao nessa funcao, verificar estado de instancias
	public void buildSets(){
		trainSet = new Instances(pool);
		testSet = new Instances(pool);
		constraintRejectedSet = new Instances(pool);
		constraintConfirmedSet = new Instances(pool);
		trainSet.delete();
		testSet.delete();
		constraintRejectedSet.delete();
		constraintConfirmedSet.delete();
		for (int i = 0; i < numInstances; i++){
			Instance instance = pool.instance(i);
			int instanceID = InstancesController.getInstanceAttributeID(instance);
			if (trainingExamplesIDs.contains((Integer)(instanceID)))
				trainSet.add(instance);
			else
				testSet.add(instance);
			
			if (constraintRejectedIDs.contains((Integer)(instanceID)))
				constraintConfirmedSet.add(instance);
			else
				if (constraintRejectedIDs.contains((Integer)(instanceID)))
					constraintRejectedSet.add(instance);
		}
		
		trainSet.setClassIndex(numAttributes-1);
		testSet.setClassIndex(numAttributes-1);
		constraintRejectedSet.setClassIndex(numAttributes-1);
		constraintConfirmedSet.setClassIndex(numAttributes-1);
	}
	
	public List<Integer> getConstraintRejectedIDs(){	
		return constraintRejectedIDs;
	}
	
	public List<Integer> getConstraintConfirmedIDs(){	
		return constraintConfirmedIDs;
	}
	
	public Instances getConstraintRejectedSet(){ 
		return constraintRejectedSet;
	}
	
	public void addConstraintRejectedIDs(List<Integer> instancesIDs){
		for (int instanceID: instancesIDs){
			addConstraintRejectedInstance(instanceID);
		}
	}
	
	public void addConstraintConfirmedIDs(List<Integer> instancesIDs){
		for (int instanceID: instancesIDs){
			addConstraintConfirmedInstance(instanceID);
		}
	}
	
	public void addConstraintRejectedInstance(int instanceID){
		if (!constraintRejectedIDs.contains(instanceID)&&(instanceID!=-1)){
			constraintRejectedIDs.add(instanceID);
			if (predictedTrueMatchingsIDs.contains(instanceID)&&(instanceID!=-1)) predictedTrueMatchingsIDs.remove((Integer)(instanceID));
			if (constraintConfirmedIDs.contains(instanceID)&&(instanceID!=-1)) constraintConfirmedIDs.remove((Integer)(instanceID));
		}
	}

	public void addConstraintConfirmedInstance(int instanceID){
		if (!constraintConfirmedIDs.contains(instanceID)&&(instanceID!=-1)){
			constraintConfirmedIDs.add(instanceID);
			if (predictedTrueMatchingsIDs.contains(instanceID)&&(instanceID!=-1)) predictedTrueMatchingsIDs.remove((Integer)(instanceID));
			if (constraintRejectedIDs.contains(instanceID)&&(instanceID!=-1)) constraintRejectedIDs.remove((Integer)(instanceID));
		}
	}

	@Override
	public void addPredictedTrueMatchingInstance(int instanceID){
		if (constraintRejectedIDs.contains(instanceID))
			System.err.printf("Trying to predict instance as TRUE but it was already rejected by network constraints. [instanceID = %d]\n",instanceID);
		else{
			if (!predictedTrueMatchingsIDs.contains(instanceID)) predictedTrueMatchingsIDs.add(instanceID);
			if (testExamplesIDs.contains(instanceID)) testExamplesIDs.remove((Integer)(instanceID));
		}
	}
	
	@Override
	public void addTrainingInstance(int instanceID){
		if (!trainingExamplesIDs.contains(instanceID)&&(instanceID!=-1)){
			trainingExamplesIDs.add(instanceID);
			if (testExamplesIDs.contains(instanceID)&&(instanceID!=-1)) testExamplesIDs.remove((Integer)(instanceID));
		}
	}

	@Override
	public void addTestInstance(int instanceID) {
		if (!testExamplesIDs.contains(instanceID)&&(instanceID!=-1)){
			testExamplesIDs.add(instanceID);
			//if (constraintConfirmedIDs.contains(instanceID)&&(instanceID!=-1)) constraintConfirmedIDs.remove((Integer)(instanceID));
			//if (constraintRejectedIDs.contains(instanceID)&&(instanceID!=-1)) constraintRejectedIDs.remove((Integer)(instanceID));
			if (trainingExamplesIDs.contains(instanceID)&&(instanceID!=-1)) trainingExamplesIDs.remove((Integer)(instanceID));
		}
	}
	
	public boolean isConstraintRejectedInstance(int instanceID){
		return constraintRejectedIDs.contains(instanceID);
	}
	
	public boolean isConstraintConfirmedInstance(int instanceID){
		return constraintConfirmedIDs.contains(instanceID);
	}
	
//	public void addTrainingInstance(String instanceString){
//		int instanceID = getMatchingCandidateID(instanceString);
//		if (instanceID!=-1) addTrainingInstance(instanceID);
//	}
	
//	public void addPredictedTrueMatchingInstance(String instanceString){
//		int instanceID = getMatchingCandidateID(instanceString);
//		if (instanceID!=-1) addPredictedTrueMatchingInstance(instanceID);
//	}
	
//	public void addConstraintConfirmedInstance(String instanceString){
//		int instanceID = getMatchingCandidateID(instanceString);
//		if (instanceID!=-1)	addConstraintConfirmedInstance(instanceID);
//	}
	
//	public void addConstraintRejectedInstance(String instanceString){
//		int instanceID = getMatchingCandidateID(instanceString);
//		if (instanceID!=-1)	addConstraintRejectedInstance(instanceID);
//	}
	
//	public void addTestInstance(String instanceString){
//		int instanceID = getMatchingCandidateID(instanceString);
//		if (instanceID!=-1) addTestInstance(instanceID);
//	}
	
	@Override
	public void printSetsInfo(){
		System.out.printf("Training set size:\t%d\n"
				+ "Test set size:    \t%d\n"
				+ "Blocked set size: \t%d\n"
				+ "Derived set size: \t%d\n",trainingExamplesIDs.size(),testSet.numInstances(),constraintRejectedIDs.size(),constraintConfirmedIDs.size());
	}

//	@Deprecated
//	/**
//	 * Returns the hash codes from the matching candidate elements. <br>
//	 * Array: SSA | EEA | SSB | EEB <br>
//	 * It returns codes in the form SSSEEE. <b>SSS</b> is the schema hash code and <b>EEE</b> is the element hash code. 
//	 */
//	public static int[] getInstanceElementsCodes(String matchingCandidateString){
//		int[] parts = new int[4];
//		String candpt1 = matchingCandidateString.split("-")[0];
//		String candpt2 = matchingCandidateString.split("-")[1];
//		parts[0] = Integer.parseInt(candpt1.split("\\[")[1].split("\\]")[0]);		//source schema code 
//		parts[1] = Integer.parseInt(candpt1.split("\\[")[0]); 						//source element code
//		parts[2] = Integer.parseInt(candpt2.split("\\[")[1].split("\\]")[0]);		//target schema code
//		parts[3] = Integer.parseInt(candpt2.split("\\[")[0]);						//target element code
//		return parts;
//		// EEE[SSS]-EEE[SSS]   : code format
//	}

	public static String getMatchingCandidateLabel(int elementUnifiedCode1, int elementUnifiedCode2){
		if (elementUnifiedCode2 < elementUnifiedCode1){						//guarantees that schemaID1 is lower than schemaID2
			int aux;
			aux = elementUnifiedCode1;		elementUnifiedCode1 = elementUnifiedCode2;	elementUnifiedCode2 = aux;
		}
		int schema1ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode1);
		int schema2ID = MatchingNetworksInstancesController.getSchemaIDFromUnifiedCode(elementUnifiedCode2);
		int element1ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode1);
		int element2ID = MatchingNetworksInstancesController.getElementIDFromUnifiedCode(elementUnifiedCode2);
		
		String cand = element1ID + "[" + schema1ID + "]-" + element2ID + "[" + schema2ID + "]";
		return cand;
	}
	
	
	
	
	
	
	

	

	

	
	
//	public MatchingNetworksInstancesController(String arffFilePath, int qtSchemas, SchemasController sc){
//		BufferedReader reader;
//		original = null;
//		try {
//			reader = new BufferedReader(new FileReader(new File(arffFilePath)));
//			original = new Instances(reader);
//			reader.close();
//		} catch (FileNotFoundException e) {
//			System.err.printf("ARFF file not found.\n");
//		} catch (IOException e) {
//			System.err.printf("Error reading ARFF file.\n");
//		}
//		matchingCandidates = new HashMap<Integer, String>();
//		candidateInstances = new HashMap<String, Integer>();
//		numInstances = original.numInstances();
//		maxElements = new int[qtSchemas];
//		for (int i=0; i<qtSchemas; i++) maxElements[i] = 0;
//		for (int i=0; i<numInstances; i++){
//			Instance inst = original.instance(i);
//			String candidate = inst.toString(1);
//			matchingCandidates.put(i, candidate);		//matching candidate attribute
//			candidateInstances.put(candidate, i);
//			String[] candParts = candidate.split("-");
//			int element1ID = Integer.parseInt(candParts[0].split("\\[")[0]);
//			int schema1ID = Integer.parseInt(candParts[0].split("\\[")[1].split("\\]")[0]);
//			int element2ID = Integer.parseInt(candParts[1].split("\\[")[0]);
//			int schema2ID = Integer.parseInt(candParts[1].split("\\[")[1].split("\\]")[0]);
//			sc.addElement(schema1ID, element1ID);
//			sc.addElement(schema2ID, element2ID);
//			if (maxElements[schema1ID] < element1ID) maxElements[schema1ID] = element1ID;
//			if (maxElements[schema2ID] < element2ID) maxElements[schema2ID] = element2ID;
//		}
//		pool = new Instances(original);
//		pool.deleteAttributeAt(1);								//remove matching candidate attribute
//		numAttributes = pool.instance(0).numAttributes();
////		System.out.printf("Numero de atributos: %d\n",numAttributes);
//		pool.setClassIndex(numAttributes-1);
//		trainingIDs = new ArrayList<Integer>();
//		testIDs = new ArrayList<Integer>();
//		blockedIDs = new ArrayList<Integer>();
//		derivedIDs = new ArrayList<Integer>();
//		labeledIDs = new ArrayList<Integer>();
//	}
//	
		
}
