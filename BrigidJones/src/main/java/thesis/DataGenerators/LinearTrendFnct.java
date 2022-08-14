package thesis.DataGenerators;

import thesis.Tools.ArrayUtils;

import java.util.HashMap;

public class LinearTrendFnct implements Function {
    @Override
    public double[] apply(HashMap<String, Object> input) {
        double power = (double)input.get("power");
        double displacement = (double)input.get("displacement");
        int timesteps = (int)input.get("timeSteps");
        int tStart = 0;
        if(input.containsKey("tStart")){
            tStart = (int)input.get("tStart");
        }
        return ArrayUtils.add(ArrayUtils.pow(ArrayUtils.range(tStart,timesteps),power),displacement);
    }
}
