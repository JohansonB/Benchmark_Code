package thesis.Models;

import java.util.ArrayList;
import java.util.HashMap;

public class TFTModel extends DartsModel {
    public TFTModel(){}

    @Override
    protected boolean is_nn() {
        return true;
    }

    public TFTModel(HashMap<String, ArrayList<String>> input){
        super(input);
    }
    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {


    }
    @Override
    public String toString(){
        return "TFT";
    }
}
