package thesis.CorrectnesChecks.oarima;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import thesis.DataGenerators.ARMA;
import thesis.Models.ARIMA_Online;
import thesis.Tools.ArrayUtils;
import thesis.Tools.GridSearcher;
import thesis.Tools.TimeSeries;
import thesis.Tools.Tuple;

import java.io.IOException;
import java.util.HashMap;

public class ARMAChangingCoefs {
    //works
    public static void main(String[] args) throws IOException {
        Tuple<double[],double[]> out =  create_Series();
        new TimeSeries(out.getVal1()).writeToCSV("Datasets\\ARMAtest2.csv");

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
        for(int i = 0; i<10;i++) {
            System.out.println(i);
            ARIMA_Online a = new ARIMA_Online(GridSearcher.reformat(h));
            a.test_accuracy("Datasets\\ARMAtest2.csv", 0.98);
            //System.out.println(m.getW());
            ARIMA_Online b = new ARIMA_Online(GridSearcher.reformat(h2));
            b.test_accuracy("Datasets\\ARMAtest2.csv", 0.98);
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
        new TimeSeries(aveA).plot("ONS");
        new TimeSeries(aveB).plot("OGD");
        new TimeSeries(aveC).plot("perfect");
    }

    private static Tuple<double[],double[]> create_Series(){
        int p = 5;
        int q = 2;
        int prevPoints = 5;
        int timeSteps = 10000;
        double[] maLags = new double[]{0.32,-0.2};
        double[] arLags;
        RealVector a1 = MatrixUtils.createRealVector(new double[]{-0.4,-0.5,0.4,0.4,0.1});
        RealVector a2 = MatrixUtils.createRealVector(new double[]{0.6,-0.4,0.4,-0.5,0.4});
        double[] startingArray = new double[]{0,0,0,0,0};
        int maxLags = Math.max(p,q);
        double[] outputArray = new double[timeSteps+maxLags];
        System.arraycopy(startingArray,startingArray.length-maxLags,outputArray,0,maxLags);
        double[] errorArray = ArrayUtils.unifrom(-0.5,0.5,timeSteps+maxLags);
        double[] meanArray = new double[timeSteps+maxLags];
        for(int i = maxLags; i<outputArray.length;i++){
            arLags = a1.mapMultiply((i+1-maxLags)/10000).add(a2.mapMultiply(1-((i+1-maxLags)/10000))).toArray();
            double value = 0;
            for(int j = 1; j<=p;j++){
                value+= outputArray[i-j]*arLags[p-j];
            }
            for(int k = 1; k<=q;k++){
                value+= errorArray[i-k]*maLags[q-k];
            }
            outputArray[i] = value+errorArray[i];
            meanArray[i] = value;
        }
        return new Tuple(ArrayUtils.startAt(outputArray,maxLags),ArrayUtils.startAt(meanArray,maxLags));



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
