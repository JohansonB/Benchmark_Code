package thesis.DataGenerators;

public class Harmonics {
    public static double[] generate(double[] sineCoef,double[]sinePeriods,double[] cosineCoef, double[] cosinePeriods, int timeSteps, int tStart){
        double[] outputArray = new double[timeSteps];
        double T = timeSteps;
        for(int i = tStart; i<timeSteps;i++){
            double value = 0;
            for(int j = 0; j<sineCoef.length;j++){
                value+= sineCoef[j]*Math.sin(i*sinePeriods[j]*2*Math.PI/T);
            }
            for(int k = 0; k<cosineCoef.length;k++){
                value+= cosineCoef[k]*Math.cos(i*cosinePeriods[k]*2*Math.PI/T);
            }
            outputArray[i] = value;
        }
        return outputArray;
    }
}
