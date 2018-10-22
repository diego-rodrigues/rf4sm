package experiments.network;

import java.util.Locale;

import structures.schema.MatchesController;
import structures.schema.MatchingNetworksInstancesController;
import experiments.Definitions;

public class TestsInconsistencies {

	public static void main(String[] args) {
		Locale.setDefault(Locale.ENGLISH);
		String domain;
		int DOMAIN_ID;
		DOMAIN_ID = Definitions.BETTING;
		DOMAIN_ID = Definitions.BUSINESS;
		DOMAIN_ID = Definitions.MAGAZINE;
		DOMAIN_ID = Definitions.BOOK;
//		DOMAIN_ID = Definitions.ORDER;
		
		domain = Definitions.DOMAIN_LIST[DOMAIN_ID];
		MatchingNetworksInstancesController instancesController;
		MatchesController matchesController;
		String arffFileName = Definitions.EXPERIMENTS + "ARFF/COMA-AVG/MatchingNetwork-" + domain + "-COMA_AvgMatcher.arff";

		instancesController = new MatchingNetworksInstancesController(arffFileName, Definitions.QT_SCHEMAS[DOMAIN_ID]);
		matchesController = new MatchesController(Definitions.QT_SCHEMAS[DOMAIN_ID]);

		//--------------------------- tests ----------------------------
		//book 0[0]-0[1] : 0 alpha
		//book 1[0]-0[1] : 43
		//book 1[0]-1[1] : 44
		//book 0[0]-0[2] : 1634 delta
		//book 0[0]-0[3] : 2546 gamma
		//book 1[0]-1[3] : 2588
		//book 0[1]-0[2] : 4104
		//book 1[1]-0[2] : 4128
		//book 0[1]-0[3] : 5136
		//book 0[1]-1[3] : 5137 beta
		//book 1[1]-1[3] : 5178
		//book 0[2]-0[3] : 6899 
		//book 0[2]-1[3] : 6900

		int code;

		matchesController.forceAddMatch(0, instancesController, false);
//		matchesController.forceAddMatch(1, instancesController, false);
//		matchesController.forceAddMatch(2, instancesController, false);
//		matchesController.forceAddMatch(43, instancesController, false);
//		matchesController.forceAddMatch(44, instancesController, false);
		matchesController.forceAddMatch(2546, instancesController, false);
//		matchesController.forceAddMatch(2588, instancesController, false);
		matchesController.forceAddMatch(4128, instancesController, false);
		matchesController.forceAddMatch(5136, instancesController, false);
//		matchesController.forceAddMatch(5137, instancesController, false);
		matchesController.forceAddMatch(5178, instancesController, false);
//		matchesController.forceAddMatch(6899, instancesController, false);
		matchesController.forceAddMatch(6900, instancesController, false);

		System.out.printf("======================================================\n");
		int insertID = 5137;
		code = matchesController.canAddMatch(insertID, instancesController);
		System.out.printf("Insert: %d\n",insertID);
		System.out.println("Conflicting Matches: " + matchesController.getLastConflictingMatches().toString());
		System.out.printf("Inconsistency code: %d\n",code);
		System.exit(0);
	}
}
