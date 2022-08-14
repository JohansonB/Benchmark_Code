package thesis.DataGenerators;

import thesis.Tools.ArrayUtils;

import java.util.HashMap;

public class NegExpTrendFct implements Function{
    @Override
    public double[] apply(HashMap<String, Object> input) {
        double dampening = (double)input.get("dampening");
        double displacement = (double)input.get("displacement");
        int timeSteps = (int)input.get("timeSteps");
        int[] steps = ArrayUtils.range(0, -1*timeSteps);
        ArrayUtils.mult(steps,dampening);
        return ArrayUtils.add(ArrayUtils.exp(steps),displacement);
    }
}
