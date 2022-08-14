package thesis.Models;

import thesis.TSModel;
import thesis.Tools.GridSearcher;
import thesis.Tools.MatrixTools;
import thesis.Tools.TimeSeries;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class TCNModel extends DartsModel {
    public TCNModel(HashMap<String, ArrayList<String>> input) {
        super(input);
    }

    public TCNModel() {

    }

    @Override
    protected boolean is_nn() {
        return true;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments+= in.containsKey("-kernel_size") ? "--kernel_size "+new Double(in.get("-kernel_size").get(0)).intValue()+" " : "";
        additionalArguments+= in.containsKey("-num_filters") ? "--num_filters "+new Double(in.get("-num_filters").get(0)).intValue()+" " : "";
        additionalArguments+= in.containsKey("-dilation_base") ? "--dilation_base "+new Double(in.get("-dilation_base").get(0)).intValue()+" " : "";

    }
    public String toString(){
        return "TCNM";
    }



}
