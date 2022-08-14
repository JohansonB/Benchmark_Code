package thesis.Models;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import thesis.TSModel;
import thesis.Tools.GridSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class TransformerModel extends DartsModel {
    private static final int default_output_length = 30;
    public TransformerModel(HashMap<String, ArrayList<String>> input) {
        super(input);
    }

    public TransformerModel() {

    }

    @Override
    protected boolean is_nn() {
        return true;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments += in.containsKey("-num_layers") ? " --num_layers "+new Double(in.get("-num_layers").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-dim_feedforward") ? " --dim_feedforward "+new Double(in.get("-dim_feedforward").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-nhead") ? " --nhead "+new Double(in.get("-nhead").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-d_model") ? " --d_model "+new Double(in.get("-d_model").get(0)).intValue() : "";
        input_chunk = in.containsKey("-output_len") ? new Double(in.get("-output_len").get(0)).intValue() : default_output_length;


    }
    public String toString(){
        return "TransformerModel";
    }


}
