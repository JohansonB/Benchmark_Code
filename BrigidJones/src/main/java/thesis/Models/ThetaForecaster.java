package thesis.Models;

import thesis.Tools.AutoCorrelation;
import thesis.Tools.Graph;
import thesis.Tools.GridSearcher;
import thesis.Tools.TimeSeries;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class ThetaForecaster extends DartsModel {

    public ThetaForecaster() {
        super(new HashMap<>());
    }

    @Override
    protected boolean is_nn() {
        return false;
    }

    public ThetaForecaster(HashMap<String, ArrayList<String>> in) {
        super(in);
    }


    public void setStepSize(int stepSize){
        super.setStepSize(stepSize);

    }

    @Override
    public String toString() {
            return "Theta";
        }

    protected void parse(HashMap<String, ArrayList<String>> input) {
        update(input);
        additionalArguments = "";
        additionalArguments += input.containsKey("-theta") ? " --theta "+new Double(input.get("-theta").get(0)).intValue() : "";
        additionalArguments += input.containsKey("-period") ? " --periods "+new Double(input.get("-period").get(0)).intValue() : "";



    }




}
