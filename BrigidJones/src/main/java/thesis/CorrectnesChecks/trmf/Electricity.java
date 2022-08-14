package thesis.CorrectnesChecks.trmf;

import thesis.Models.trmf;
import thesis.Tools.GridSearcher;
import thesis.Tools.MatrixTools;
import thesis.Tools.TimeSeries;

import java.io.IOException;
import java.util.HashMap;

public class Electricity {
    public static void main(String[] args) throws IOException {
        HashMap<String,String> h = new HashMap<>();
        h.put("-lambdaf","2");
        h.put("-lambdaw","625");
        h.put("-lambdax","0.5");
        h.put("-windowSize","24");
        TimeSeries ts2 = new TimeSeries("Datasets\\trafficTRMF.csv");
        TimeSeries ts = new TimeSeries("Datasets\\electriciryTRMF.csv");
        double k = 40.0/ts2.getDimension();
        h.put("-k","40");
        trmf model = new trmf(GridSearcher.reformat(h));
        double split = 1-((double)7*24)/ts.length();
        //System.out.println(MatrixTools.getDimensions(ts.toMatrix()));
        //System.out.println(MatrixTools.getDimensions(ts2.toMatrix()));

        model.test_accuracy("Datasets\\electriciryTRMF.csv",split).plotPairwiseComparrison(10);


    }
}
