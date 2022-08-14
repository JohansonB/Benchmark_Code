package thesis.CorrectnesChecks.oarima;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import thesis.DataGenerators.ARMA;
import thesis.Models.ARIMA_Online;
import thesis.TSModel;
import thesis.Tools.GridSearcher;
import thesis.Tools.TimeSeries;
import thesis.Tools.Tuple;

import java.io.IOException;
import java.util.HashMap;

/*
Test correct both methods RMSE converge towards the RMSE of the perfect predictor
 */

public class SanityCheck {
    public static void main(String[] args) throws IOException {
        Tuple<double[],double[]> out =  ARMA.generate(new double[]{0.6,-0.5,0.4,-0.4,0.3},new double[]{0.3,-0.2},new double[]{0,0,0,0,0},10000,0,0.3);
        new TimeSeries(out.getVal1()).writeToCSV("Datasets\\ARMA.csv");

        HashMap<String,String> h =new HashMap<>();
        h.put("-opt","ONS");
        h.put("-epsilon","0.1");
        h.put("-lrate","1");
        HashMap<String,String> h2 =new HashMap<>();
        h2.put("-opt","OGD");
        h2.put("-lrate","1");
        RealVector aveA = null;
        RealVector aveB = null;
        RealVector aveC = null;
        double RMSEA = 0;
        for(int i = 0; i<10;i++) {
            System.out.println(i);
            ARIMA_Online a = new ARIMA_Online(GridSearcher.reformat(h));
            RMSEA += a.test_accuracy("Datasets\\ARMA.csv", 0.98).error(new TSModel.RMSE());

            //System.out.println(m.getW());
            ARIMA_Online b = new ARIMA_Online(GridSearcher.reformat(h2));
            b.test_accuracy("Datasets\\ARMA.csv", 0.98);
            //System.out.println(b.getW());
            if(aveA == null)
                aveA = a.getRMSE();
            else
                aveA = aveA.mapMultiply(i+1).add(a.getRMSE()).mapDivide(i+2);
            if(aveB == null)
                aveB = b.getRMSE();
            else
                aveB = aveB.mapMultiply(i+1).add(b.getRMSE()).mapDivide(i+2);
            if(aveC==null)
                aveC = getRMSE(out);
            else
                aveC = aveC.mapMultiply(i+1).add(getRMSE(out)).mapDivide(i+2);
        }
        RMSEA /= 10;
        System.out.println(RMSEA);
        new TimeSeries(aveA).plot("ONS");
        new TimeSeries(aveB).plot("OGD");
        new TimeSeries(aveC).plot("perfect");
    }
    private static RealVector getRMSE(Tuple<double[],double[]> out) {
        RealVector a = MatrixUtils.createRealVector(out.getVal1());
        RealVector b = MatrixUtils.createRealVector(out.getVal2());
        RealVector RMSE = MatrixUtils.createRealVector(new double[a.getDimension()]);
        double SE = 0;
        for(int i = 0; i<a.getDimension();i++){
            SE += Math.pow(a.getEntry(i)-b.getEntry(i),2);
            RMSE.setEntry(i,Math.sqrt(SE/(i+1)));
        }
        return RMSE;
    }
}
