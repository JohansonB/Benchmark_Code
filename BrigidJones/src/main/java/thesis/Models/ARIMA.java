package thesis.Models;

import java.util.ArrayList;
import java.util.HashMap;

public class ARIMA extends DartsModel {
    private boolean auto = true;
    final static int defaultP = 2;
    final static int defaultD = 0;
    final static int defaultQ = 2;


    public ARIMA() {
        super(new HashMap<>());
    }

    @Override
    protected boolean is_nn() {
        return false;
    }

    @Override
    public void setStepSize(int stepSize){
        super.setStepSize(stepSize);
    }

    public ARIMA(HashMap<String, ArrayList<String>> in) {
        super(in);
    }


    int p;
    int d;
    int q;


    protected void parse(HashMap<String, ArrayList<String>> input) {
        update(input);
            p = input.containsKey("-p") ? new Integer(input.get("-p").get(0)) : defaultP;
            d = input.containsKey("-d") ? new Integer(input.get("-d").get(0)) : defaultD;
            q = input.containsKey("-q") ? new Integer(input.get("-q").get(0)) : defaultQ;
            additionalArguments = "--p " + p + " --d " + d + " " + "--q " + q;
    }


    public String toString() {
        if (!auto)
            return "ARIMA";
        else
            return "AutoARIMA";
    }

    public void setAuto(boolean b) {
        auto = b;
    }



}
