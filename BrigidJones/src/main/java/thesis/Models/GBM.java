package thesis.Models;

import thesis.Tools.GridSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class GBM extends DartsModel {
    private static final int default_lag = 100;
    private int lag;
    public GBM(){
        super();
    }
    public GBM(HashMap<String, ArrayList<String>> in){
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
        additionalArguments += in.containsKey("-lag") ? " --lag "+ new Double(in.get("-lag").get(0)).intValue() : "";
        lag = in.containsKey("-lag") ?  new Double(in.get("-lag").get(0)).intValue() : default_lag;
        additionalArguments += in.containsKey("-max_depth") ? " --max_depth "+ new Double(in.get("-max_depth").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-n_estimators") ? " --n_estimators "+ new Double(in.get("-n_estimators").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-output_chunk_length") ? " --output_chunk_length "+ new Double(in.get("-output_chunk_length").get(0)).intValue() : "";
    }
    @Override
    public boolean isND(){
        return true;
    }

    @Override
    protected void update_params_context() {
        if(o.getTrain().getColumnDimension()<=lag+1){
            lag = (int)(o.getTrain().getColumnDimension()*0.1);
            additionalArguments += " --lag "+lag;
        }
    }

    public String toString(){
        return "GBM";
    }


}
