package thesis.Models;

import java.util.ArrayList;
import java.util.HashMap;

public class SES extends DartsModel {

    public SES() {
        super(new HashMap<>());
    }

    @Override
    protected boolean is_nn() {
        return false;
    }

    public SES(HashMap<String, ArrayList<String>> in) {
        super(in);
    }


    public void setStepSize(int stepSize){
        super.setStepSize(stepSize);
    }




    private Integer periods;
    private static Integer defaultPeriods = null;



    @Override
    public String toString() {
            return "ETS";
        }

    protected void parse(HashMap<String, ArrayList<String>> input) {
        additionalArguments = "";
        update(input);
        periods = input.containsKey("-periods") ? new Integer(input.get("-periods").get(0)) : defaultPeriods;
        additionalArguments += periods != null ? " --periods "+periods : "";
        additionalArguments += input.containsKey("-trend") ? " --trend_ETS "+input.get("-trend").get(0) : "";
    }


}
