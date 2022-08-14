package thesis.DataGenerators;

import java.util.HashMap;

public class TrendSeries {
    public static double[] generate(Function f, HashMap<String,Object> input){
        return f.apply(input);
    }
}
