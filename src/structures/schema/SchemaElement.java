package structures.schema;

import java.util.ArrayList;

public class SchemaElement {
	
	private int schemaID;
	private int elementID;
	private ArrayList<Integer> matches;
	
	public SchemaElement(int schemaID, int elementID){
		this.schemaID = schemaID;
		this.elementID = elementID;
		matches = new ArrayList<Integer>();
	}
	
	public SchemaElement(int schemaElementID){
		this.schemaID = InstancesController.getSchemaIDFromUnifiedCode(schemaElementID);
		this.elementID = InstancesController.getElementIDFromUnifiedCode(schemaElementID);
		matches = new ArrayList<Integer>();
	}
	
	public SchemaElement clone(){
		SchemaElement clone = new SchemaElement(schemaID,elementID);
		for (int match: matches) clone.addMatch(match);
		return clone;
	}
	
	public int getSchemaID(){ return schemaID;}
	public int getElementID(){ return elementID;}
	
	public void addMatch(int matchID){
		if (!matches.contains(matchID)) matches.add(matchID);
	}
	
	public void removeMatch(int matchID){
		if (matches.contains(matchID)) matches.remove((Integer)matchID);
	}
	
	/**
	 * Verifies if this element already contains a match to the specified schema. (inconsistency type 1)
	 * @return the elements IDs from the correspondent schema this element is matched. Return -1 if no match is 
	 * identified.
	 */
	public ArrayList<Integer> containSchema(int schemaElementID){
		int schemaID = InstancesController.getSchemaIDFromUnifiedCode(schemaElementID);
		ArrayList<Integer> inconsistencies = new ArrayList<Integer>();
		for (Integer matchID: matches){
			if (InstancesController.getSchemaIDFromUnifiedCode(matchID) == schemaID)
				if (matchID != schemaElementID) 
					inconsistencies.add(matchID);
		}
		if (inconsistencies.size()>0) return inconsistencies;
		else return null;
	}
	
	/**
	 * Returns the size of the matches list of this schema element. 
	 */
	public int qtMatches(){return matches.size();}
	
	public ArrayList<Integer> getMatches(){return matches;}
	
	public boolean isMatched(int schemaID, int elementID){
		int matchElementID = InstancesController.getUnifiedCode(schemaID, elementID);
		return matches.contains(matchElementID);
	}
	
	public boolean isMatched(int schemaElementID){
		return matches.contains(schemaElementID);
	}
	
	public void printMatches(){
		System.out.printf("Element #%06d\t-> ",InstancesController.getUnifiedCode(schemaID, elementID));
		for (int a: matches) System.out.printf("#%06d  ",a);
		System.out.printf("\n");
	}
}
