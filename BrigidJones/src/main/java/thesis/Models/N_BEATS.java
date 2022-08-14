package thesis.Models;

import thesis.Tools.Graph;
import thesis.Tools.GridSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class N_BEATS extends DartsModel{

    public N_BEATS(HashMap<String, ArrayList<String>> in) {
        super(in);
    }
    public N_BEATS() {
        super();
    }

    @Override
    protected boolean is_nn() {
        return true;
    }

    @Override
    public boolean isND(){
        return false;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments+= in.containsKey("-num_stacks") ? " --num_stacks "+new Double(in.get("-num_stacks").get(0)).intValue() : "";
        additionalArguments+= in.containsKey("-num_layers") ? " --num_layers_N_BEATS "+new Double(in.get("-num_layers").get(0)).intValue() : "";
        additionalArguments+= in.containsKey("-layer_widths") ? " --layer_widths "+new Double(in.get("-layer_widths").get(0)).intValue() : "";



    }

    public String toString(){
        return "NBEATS";
    }


}
