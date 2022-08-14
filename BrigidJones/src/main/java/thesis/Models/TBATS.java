package thesis.Models;

import java.util.ArrayList;
import java.util.HashMap;

public class TBATS extends DartsModel {
    public TBATS(){
        super();
    }
    public TBATS(HashMap<String,ArrayList<String>> in){
        super(in);
    }
    @Override
    protected boolean is_nn() {
        return false;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments+= in.containsKey("-periods") ? " --periods_TBATS "+in.get("-periods").get(0) : "";
        additionalArguments+= in.containsKey("-trend") ? " --trend_TBATS "+(new Boolean(in.get("-trend").get(0)).booleanValue() ? "True" : "False") : "";

    }

    @Override
    public String toString() {
        return "TBATS";
    }
}
