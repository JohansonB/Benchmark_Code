package thesis.DataGenerators;

import thesis.Tools.ArrayUtils;

import java.util.HashMap;

public class LogTrendFunction implements Function{
    @Override
    public double[] apply(HashMap<String, Object> input) {
        double displacement = (double)input.get("displacement");
        int timeSteps = (int)input.get("timeSteps");
        int sStart = 0;
        if(input.containsKey("tStart")){
            sStart = (int)input.get("tStart");
        }
        return ArrayUtils.add(ArrayUtils.log(ArrayUtils.range(sStart+1,timeSteps+1)),displacement);
    }
}
