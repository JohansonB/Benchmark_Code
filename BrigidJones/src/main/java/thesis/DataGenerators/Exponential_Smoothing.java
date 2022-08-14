package thesis.DataGenerators;

import thesis.Tools.ArrayUtils;
import thesis.Tools.TimeSeries;

public class Exponential_Smoothing {
    public static TimeSeries ATAS(double L, double R, double[] S, double SD, int T){
        double[] error = ArrayUtils.normal(0,SD,T);
        double[] result = new double[T];
        for(int i = 0; i<result.length;i++){
            result[i] = L + i*R + S[i%S.length] + error[i];
        }
        return new TimeSeries(result);
    }
    public static TimeSeries ATMS(double L, double R, double[] S, double SD, int T){
        double[] error = ArrayUtils.normal(0,SD,T);
        double[] result = new double[T];
        for(int i = 0; i<result.length;i++){
            result[i] = (L + i*R) * S[i%S.length] + error[i];
        }
        return new TimeSeries(result);
    }
    public static TimeSeries MTAS(double L, double R, double[] S, double SD, int T){
        double[] error = ArrayUtils.normal(0,SD,T);
        double[] result = new double[T];
        for(int i = 0; i<result.length;i++){
            result[i] = (L *Math.pow(R,i)) + S[i%S.length] + error[i];
        }
        return new TimeSeries(result);
    }
    public static TimeSeries MTMS(double L, double R, double[] S, double SD, int T){
        double[] error = ArrayUtils.normal(0,SD,T);
        double[] result = new double[T];
        for(int i = 0; i<result.length;i++){
            result[i] = (L *Math.pow(R,i)) * S[i%S.length] + error[i];
        }
        return new TimeSeries(result);
    }
}
