package structures.schema;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.sql.PooledConnection;

import experiments.Definitions;

import weka.core.Instance;
import weka.core.Instances;

public abstract class InstancesController {
	
	public static final int TRAIN = 1;
	public static final int TEST = 2;
	public static final int LABEL = 3;

	private HashMap<Integer, String> matchingCandidates;
	private HashMap<String, Integer> candidateInstances;

	protected Instances original;								//original is the set with all the attributes and instances
	protected Instances pool;									//pool is the set with usable attributes and valid instances
	protected Instances trainSet;
	protected Instances testSet;
	
	protected List<Integer> trainingExamplesIDs;				//the IDs of train instances
	protected List<Integer> testExamplesIDs;					//the IDs of test instances
	protected List<Integer> predictedTrueMatchingsIDs;			//instances labeled by classifiers
	
	protected int[] maxElements;								//number of elements per schema
	protected int numInstances;									//the number of instances in pool
	protected int numAttributes;								//the number of attributes in the data set
	
	public InstancesController(String arffFilePath, int qtSchemas){
		matchingCandidates = new HashMap<Integer,String>();
		candidateInstances = new HashMap<String,Integer>();
		original = null;
		pool = null;
		trainSet = null;
		trainingExamplesIDs = new ArrayList<Integer>();
		testExamplesIDs = new ArrayList<Integer>();
		predictedTrueMatchingsIDs = new ArrayList<Integer>();
		
		BufferedReader reader;
		original = null;
		try {
			reader = new BufferedReader(new FileReader(new File(arffFilePath)));
			original = new Instances(reader);
			reader.close();
		} catch (FileNotFoundException e) {
			System.err.printf("ARFF file not found.\n");
		} catch (IOException e) {
			System.err.printf("Error reading ARFF file.\n");
		}
		numInstances = original.numInstances();
		maxElements = new int[qtSchemas];
		for (int i=0; i<qtSchemas; i++) maxElements[i] = 0;
		for (int instID=0; instID<numInstances; instID++){
			Instance inst = original.instance(instID);
			String candidate = inst.toString(1);
			matchingCandidates.put(instID, candidate);		//matching candidate attribute
			candidateInstances.put(candidate, instID);
			int[] codes = getInstanceElementsCodes(instID);
			int schemaA = codes[0];
			int elementA = codes[1];
			int schemaB = codes[2];
			int elementB = codes[3];
			addTestInstance(instID);
			if (maxElements[schemaA] < elementA) maxElements[schemaA] = elementA;
			if (maxElements[schemaB] < elementB) maxElements[schemaB] = elementB;
		}
		pool = new Instances(original);
		pool.deleteAttributeAt(1);								//remove matching candidate attribute
		numAttributes = pool.instance(0).numAttributes();
		pool.setClassIndex(numAttributes-1);
	}
	
	/**
	 * Returns the hash codes from the matching candidate elements. <br>
	 * It returns an array of codes in the form SSA | EEA | SSB | EEB.<br> 
	 * <b>SSA</b> is the source schema hash code. <br>
	 * <b>SSB</b> is the target schema hash code. <br>
	 * <b>EEA</b> and <b>EEB</b> are the elements hash codes from the respectively schemas.<br> 
	 */
	protected abstract int[] getInstanceElementsCodes(int instanceID);
	
	/**
	 * Build the sets of instances according to previously divided sets lists.
	 */
	public abstract void buildSets();
	
	public abstract void addTrainingInstance(int instanceID);
	
	public abstract void addTestInstance(int instanceID);
	
	public abstract void addPredictedTrueMatchingInstance(int instanceID);
	
	public abstract void printSetsInfo();
	
	public void addTrainIDs(List<Integer> instancesIDs){
		for (int instID:instancesIDs)
			addTrainingInstance(instID);
	}
	
	public void addTestIDs(List<Integer> instancesIDs){
		for (int instID:instancesIDs)
			addTestInstance(instID);
	}
	
	public void addPredictedTrueMatchingsIDs(List<Integer> instancesIDs){
		for (int instID:instancesIDs)
			addPredictedTrueMatchingInstance(instID);
	}
	
	/**
	 * Returns the ID attribute of the instance.
	 */
	protected int getMatchingCandidateID(String matchingCandidate){
		int matchID;
		if (candidateInstances.containsKey(matchingCandidate))
			matchID = candidateInstances.get(matchingCandidate);
		else matchID = -1;
		return matchID;
	}
	
	@Deprecated
	/**
	 * Returns the ID attribute of the instance.
	 */
	public int getMatchingCandidateID(int schemaID1, int elementID1, int schemaID2, int elementID2){
		String matchingCandidate = getMatchingCandidateLabel(schemaID1, elementID1, schemaID2, elementID2);
		int matchID;
		if (candidateInstances.containsKey(matchingCandidate))
			matchID = candidateInstances.get(matchingCandidate);
		else matchID = -1;
		return matchID;
	}
	
	/**
	 * Returns the ID attribute of the instance.
	 */
	public int getMatchingCandidateID(int schemaElementID1, int schemaElementID2){
		String matchingCandidate = getMatchingCandidateLabel(getSchemaIDFromUnifiedCode(schemaElementID1),
															 getElementIDFromUnifiedCode(schemaElementID1),
															 getSchemaIDFromUnifiedCode(schemaElementID2),
															 getElementIDFromUnifiedCode(schemaElementID2));
		return getMatchingCandidateID(matchingCandidate);
	}
	
	/**
	 * Given schema and elements codes, it returns the candidate string.
	 */
	private static String getMatchingCandidateLabel(int schemaID1, int elementID1, int schemaID2, int elementID2){
		if (schemaID1 > schemaID2){
			int aux = elementID1;
			elementID1 = elementID2;
			elementID2 = aux;
			aux = schemaID1;
			schemaID1 = schemaID2;
			schemaID2 = aux;
		}
		String cand = elementID1 + "[" + schemaID1 + "]-" + elementID2 + "[" + schemaID2 + "]";
		return cand;
	}
	
	/**
	 * @return Returns the matching candidate string (feature #2 from original ARFF file).
	 */
	public String getMatchingCandidateLabel(int instanceAttributeID){
		if (!matchingCandidates.containsKey(instanceAttributeID)) return null;
		return matchingCandidates.get(instanceAttributeID);
	}

	public static int getInstanceAttributeID(Instance inst){
		return (int)inst.value(0);
	}

	/**
	 * Returns the class value of the instance.
	 */
	public static int getInstanceClassValue(Instance inst){
		return (int)inst.value(inst.numAttributes()-1);
	}
	
	/**
	 * Returns the class value of the instance indexed by the given ID.
	 */
	public int getInstanceClassValue(int instID){
		Instance instance = pool.instance(instID);
		return getInstanceClassValue(instance);
	}
	
	/**
	 * Returns the class value of the instance.
	 */
	public static String getInstanceStringClassValue(Instance inst){
		if (getInstanceClassValue(inst) == Definitions.TRUE)
			return "True";
		else return "False";
	}

	public int getQuantityOfElementsFromSchema(int schemaID){
		return maxElements[schemaID];
	}
	
	/**
	 * Adds a <i>true</i> instance in the pool. 
	 */
	public void addVoidTrueInstance(){
		Instance fake = (Instance) pool.firstInstance().copy();
		fake.setValue(0, original.numInstances()); //fake instance ID
		int sizeAttributes = pool.instance(0).numAttributes();
		for (int i=1; i<sizeAttributes-1; i++) fake.setValue(i, 1.00);
		fake.setValue(sizeAttributes-1, "true");
		if (!trainingExamplesIDs.contains(original.numInstances())) trainingExamplesIDs.add(original.numInstances());	//add fake instance to training
		pool.add(fake);
		numInstances++;
	}

	/**
	 * Function used to get a custom set of instances.
	 */
	public Instances getCustomSet(List<Integer> instancesIDs){
		Instances customized = new Instances(pool);
		customized.delete();
		for (int instID: instancesIDs){
			Instance inst = (Instance)pool.instance(instID).copy();
			customized.add(inst);
		}
		return customized;
	}

	public List<Integer> getPredictedTrueMatchingsIDs() {
		return predictedTrueMatchingsIDs;
	}
	
	public List<Integer> getTrainingExamplesIDs() {
		return trainingExamplesIDs;
	}
	
	public List<Integer> getTestIDs() {
		return testExamplesIDs;
	}
	
	public Instances getTrainingSet(){
		return trainSet;
	}
	
	public Instances getOriginalSet(){ 
		return original;
	}
	
	public Instances getPoolSet(){ 
		return pool;
	}
	
	public Instances getTestSet(){ 
		return testSet;
	}
	
	public boolean isTrainingInstance(int instanceID){
		return trainingExamplesIDs.contains(instanceID);
	}
	
	public boolean isTestInstance(int instanceID){
		return testExamplesIDs.contains(instanceID);
	}
	
	public boolean isPredictedTrueMatchingInstance(int instanceID){
		return predictedTrueMatchingsIDs.contains(instanceID);
	}
	
	private static final int SCHEMA_NUMBER_HASH = 1000;

	public static int getUnifiedCode(int schemaID, int elementID){
		return schemaID * SCHEMA_NUMBER_HASH + elementID;
	}
	
	public static int getSchemaIDFromUnifiedCode(int schemaElementID){
		return schemaElementID/SCHEMA_NUMBER_HASH;
	}
	
	public static int getElementIDFromUnifiedCode(int schemaElementID){
		return schemaElementID%SCHEMA_NUMBER_HASH;
	}
	
	public static void printUnifiedCode(int schemaElementID){
		System.out.printf("S%03d.E%03d",getSchemaIDFromUnifiedCode(schemaElementID),getElementIDFromUnifiedCode(schemaElementID));		
	}
	
	public static String getStringUnifiedCode(int schemaElementID){
		String a = String.format("S%03d.E%03d",getSchemaIDFromUnifiedCode(schemaElementID),getElementIDFromUnifiedCode(schemaElementID));
		return a;
	}
	
	public int getNumInstances(){
		return numInstances;
	}
}
