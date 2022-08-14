package thesis.Models;

import java.util.ArrayList;
import java.util.HashMap;

public class BATS extends DartsModel {
    public BATS(){
        super();
    }
    public BATS(HashMap<String,ArrayList<String>> in){
        super(in);
    }
    @Override
    protected boolean is_nn() {
        return false;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);

    }

    @Override
    public String toString() {
        return "BATS";
    }
}
