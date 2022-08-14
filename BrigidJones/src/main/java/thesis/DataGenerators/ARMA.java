package thesis.DataGenerators;

import thesis.Tools.ArrayUtils;
import thesis.Tools.TimeSeries;
import thesis.Tools.Triple;
import thesis.Tools.Tuple;

import java.util.Arrays;

public class ARMA {
    public static Tuple<double[],double[]> generate(double[] arLags, double[] maLags, double[] startingArray, int timeSteps, double noiseMean, double noiseSD){
        int p = arLags.length;
        int q = maLags.length;
        int prevPoints = startingArray.length;
        if(p>prevPoints){
            throw new IllegalArgumentException("starting array must be of length >= arLags");
        }
        if(q>prevPoints){
            throw new IllegalArgumentException("startingArray must be of length >= maLags");
        }
        int maxLags = Math.max(p,q);
        double[] outputArray = new double[timeSteps+maxLags];
        System.arraycopy(startingArray,startingArray.length-maxLags,outputArray,0,maxLags);
        double[] errorArray = ArrayUtils.normal(noiseMean,noiseSD,timeSteps+maxLags);
        double[] meanArray = new double[timeSteps+maxLags];
        for(int i = maxLags; i<outputArray.length;i++){
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
    public static TimeSeries sarima(){
        double[] out_arr = new double[104];
        double p = 0.16;
        double q = 0.7;
        double[] errors = ArrayUtils.normal(0,0.000000000000000000000000001,100+4);
        out_arr[0] = 5;
        out_arr[1] = 10;
        out_arr[2] =20;
        out_arr[3] = 30;
        for(int i = 4; i<104;i++){
            out_arr[i] = errors[i]+p*out_arr[i-1]+q*out_arr[i-3]+p*q*out_arr[i-4];

        }
        return new TimeSeries(out_arr);

    }

}
