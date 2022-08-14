package thesis.Models;

import thesis.Tools.GridSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class FFT extends DartsModel {
    public FFT(){

    }

    @Override
    protected boolean is_nn() {
        return false;
    }



    public FFT(HashMap<String, ArrayList<String>> in){
        super(in);
    }
    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments += in.containsKey("-trend") ? " --trend "+in.get("-trend").get(0) : "";
        additionalArguments += in.containsKey("-nb_terms") ? " --nb_terms "+ new Double(in.get("-nb_terms").get(0)).intValue() : "";
        additionalArguments += in.containsKey("-trend_poly_degree") ? " --trend_poly_degree "+new Double(in.get("-trend_poly_degree").get(0)).intValue() : "";

    }

    public String toString(){
        return "FFT";
    }


}
