package thesis.Tools;

import net.quasardb.qdb.exception.Exception;
import org.apache.commons.math3.distribution.UniformRealDistribution;

import java.util.ArrayList;
import java.util.Random;

public class ArrayUtils {
    public static double[] div(double[] a, double[] b){
        double[] ret = new double[a.length];
        for(int i = 0; i<a.length;i++){
            ret[i] = a[i]/b[i];
        }
        return ret;
    }
    public static double[] sub(double[] a, double[] b){
        double[] ret = new double[a.length];
        for(int i = 0; i<ret.length;i++){
            ret[i] = a[i]-b[i];
        }
        return ret;
    }
    public static double[] to1D(Double[][] in){
        double[] ret = new double[in.length];
        for(int i = 0; i<ret.length;i++){
            ret[i] = in[i][0];
        }
        return ret;
    }
    public static double to1D(Double[] in){
        return in[0];
    }
    public static double[][] to2D(double[] in){
        double[][] ret = new double[in.length][1];
        for(int i = 0; i<in.length;i++ ){
            ret[i][0] = in[i];
        }
        return ret;
    }
    public static Double[][] reverse(Double[][] in){
        int i = in.length-1;
        int j = 0;
        Double[] temp;
        while (i>j){
            temp = in[i];
            in[i] = in[j];
            in[j] = temp;
            i--;
            j++;
        }
        return in;
    }
    // reverses array elements  [1,2,3] -> [3,2,1]
    //changes value of input variable
    public static double[] reverse(double[] in){
        int end = in.length-1;
        int start = 0;
        double temp;
        while (start<end){
            temp = in[start];
            in[start] = in[end];
            in[end] = temp;
            start++;
            end--;
        }
        return in;
    }
    public static double[][] toPrimitive(Double[][] in){
        double[][] ret = new double[in.length][];
        for(int i = 0; i<in.length;i++){
            ret[i] = new double[in[i].length];
            for(int j = 0; j<in[i].length;j++){
                ret[i][j] = in[i][j];
            }
        }
        return ret;
    }
    public static double[] toPrimitive(Double[] in){
        double[] ret = new double[in.length];
        for(int i = 0; i<in.length;i++){
            ret[i] = in[i];
        }
        return ret;
    }
    public static double[][] transpose(double[][] in){
        double[][] ret = new double[in[0].length][in.length];
        int length = -1;
        for(int i = 0; i<in.length;i++){
            if(in[i].length>length){
                length = in[i].length;
            }
        }
        for(int i = 0; i<in.length;i++){
            if(in[i].length<length){
               double[] temp = new double[length];
               System.arraycopy(in[i],0,temp,0,in[i].length);
               for(int j = in[i].length; j<length;j++)
                   temp[j] = Double.NaN;
               in[i] = temp;

            }
        }
        for(int i = 0; i<ret.length;i++){
            for(int j = 0; j<ret[i].length;j++){
                ret[i][j] = in[j][i];
            }
        }
        return ret;
    }
    public static double[][] zeros(int n, int m){
        double[][] ret = new double[n][m];
        for(int i = 0; i<n;i++){
            for(int j = 0; j<m;j++){
                ret[i][j] = 0;
            }
        }
        return ret;
    }
    public static ArrayList<Tuple<Integer,Integer>> isNaN(double[][] in){
        ArrayList<Tuple<Integer,Integer>> ret = new ArrayList<>();
        for(int i = 0; i<in.length;i++){
            for(int j = 0; j<in[i].length;j++){
                if(Double.isNaN(in[i][j])){
                    ret.add(new Tuple<>(i,j));
                }

            }
        }
        return ret;
    }
    public static double[] to1D(double[][] in){
        double[] ret = new double[in.length];
        for(int i = 0; i<in.length;i++){
            ret[i] = in[i][0];
        }
        return ret;
    }
    public static ArrayList<Tuple<Integer,Integer>> complementIndexes(int n, int m,ArrayList<Tuple<Integer,Integer>> indexes){
        Tuple<Integer,Integer> cur;
        ArrayList<Tuple<Integer,Integer>> ret = new ArrayList<>();
        for(int i = 0; i<n;i++){
            for(int j = 0; j<m;j++){
                cur = new Tuple<>(i,j);
                if(!indexes.contains(cur)){
                    ret.add(cur);
                }
            }
        }
        return ret;
    }
    public static double[] flatten(double[][] in, boolean by_rows){
        double[] ret = new double[in.length*in[0].length];
        int count = 0;
        if(by_rows){
            for(int i = 0; i<in.length;i++){
                for(int j = 0; j<in[i].length;j++){
                    ret[count++] = in[i][j];
                }
            }
        }
        else{
            for(int j = 0; j<in[0].length;j++){
                for(int i = 0; i<in.length;i++){
                    ret[count++] = in[i][j];
                }
            }

        }
        return ret;
    }
    public static double[][] to2D(double[] in,int n, int m){
        double[][] ret = new double[n][m];
        int count = 0;
        for(int j =0;j<m;j++){
            for(int i = 0; i<n;i++){
                ret[i][j] = in[count++];
            }
        }
        return ret;

    }
    public static double[] append(double[] a, double[] b){
        double[] ret = new double[a.length+b.length];
        System.arraycopy(a,0,ret,0,a.length);
        System.arraycopy(b,0,ret,a.length,b.length);
        return ret;
    }

    public static int[] append(int[] a, int[] b){
        int[] ret = new int[a.length+b.length];
        System.arraycopy(a,0,ret,0,a.length);
        System.arraycopy(b,0,ret,a.length,b.length);
        return ret;
    }
    public static double[][] max(int i, int j){
        double[][] ret = new double[i][j];
        for(int k = 0; k<i;k++){
            for(int z=0;z<j;z++){
                ret[k][z] = Double.MAX_VALUE;
            }
        }
        return ret;
    }
    public static double[] zeros(int len){
        double[] ret = new double[len];
        for(int i = 0;i<len;i++){
            ret[i] = 0;
        }
        return ret;
    }
    public static double[] normal(double mean, double SD, int len){
        double[] ret = new double[len];
        Random r = new Random();
        for(int i = 0; i<len;i++){
            ret[i] = r.nextGaussian()*SD+mean;
        }
        return ret;
    }
    public static double[] unifrom(double lower, double upper, int len){
        double[] ret = new double[len];
        UniformRealDistribution x = new UniformRealDistribution(lower,upper);
        for(int i = 0; i<len;i++){
            ret[i] = x.sample();
        }
        return ret;
    }
    public static double[] startAt(double[] in, int start){
        double[] ret = new double[in.length-start];
        System.arraycopy(in,start,ret,0,ret.length);
        return ret;
    }
    public static int[] range(int start, int end){
        int[] ret = new int[end-start];
        if(start<=end) {
            for (int i = 0; i < ret.length; i++) {
                ret[i] = start++;

            }
        }
        else{
            for (int i = 0; i < ret.length; i++) {
                ret[i] = start--;

            }

        }
        return ret;
    }
    public static double[] pow(int[] in,double pow){
        double[] ret = new double[in.length];
        for(int i = 0; i<in.length;i++){
            ret[i] = Math.pow(in[i],pow);
        }
        return ret;
    }
    public static double[] pow(double[] in,double pow){
        double[] ret = new double[in.length];
        for(int i = 0; i<in.length;i++){
            ret[i] = Math.pow(in[i],pow);
        }
        return ret;
    }
    public static double[] add(double[] a, double b){
        for(int i = 0; i<a.length;i++){
            a[i] += b;
        }
        return a;
    }
    public static double[] mapAdd(double[] a, double b){
        double[] ret = new double[a.length];
        for(int i = 0; i<a.length;i++){
            ret[i] = a[i] + b;
        }
        return ret;
    }
    public static double[][] add(double[][] a, double b){
        for(int i = 0; i<a.length;i++){
            for(int j = 0; j<a[i].length;j++){
                a[i][j] += b;
            }
        }
        return a;
    }
    public static double[] log(int[] a){
        double[] ret = new double[a.length];
        for(int i = 0; i<a.length;i++){
            ret[i] = Math.log(a[i]);
        }
    return ret;
    }
    public static double[][] log(double[][] a){
        double[][] ret = new double[a.length][a[0].length];
        for(int i = 0; i<a.length;i++){
            for(int j = 0; j<a[i].length;j++) {
                ret[i][j] = Math.log(a[i][j]);
            }
        }
        return ret;
    }
    public static double[] mult(double[] a,double m){
        for(int i= 0;i<a.length;i++){
            a[i] *= m;
        }
        return a;

    }
    public static double[] mapMult(double[] a,double m){
        double[] ret = new double[a.length];
        for(int i= 0;i<a.length;i++){
            ret[i] = a[i]*m;
        }
        return ret;

    }
    public static double[][] mult(double[][] a,double m){
        for(int i= 0;i<a.length;i++){
            for(int j = 0; j<a[i].length;j++) {
                a[i][j] *= m;
            }
        }
        return a;

    }
    public static double[] mult(double[] a,double[] b){
        double[] ret = new double[a.length];
        for(int i= 0;i<a.length;i++){
            ret[i] = a[i]*b[i];
        }
        return ret;

    }
    public static int[] mult(int[] a,double m){
        for(int i= 0;i<a.length;i++){
            a[i] *= m;
        }
        return a;

    }
    public static double[] exp(int[] a){
        double[] ret = new double[a.length];
        for(int i = 0; i<a.length;i++){
           ret[i] = Math.exp(a[i]);
        }
        return ret;
    }
    public static double[] add(double[] a, double[] b){
        double[] ret = new double[a.length];
        for(int i = 0; i<a.length;i++){
            ret[i] = a[i]+b[i];
        }
        return ret;
    }
    public static double[] selfAdd(double[] a, double[] b){
        for(int i = 0; i<a.length;i++){
            a[i] += b[i];
        }
        return a;
    }
    public static double nanMax(double[] in){
        double max = Double.MIN_VALUE;
        for(int i = 0; i<in.length;i++){
            if(!Double.isNaN(in[i])&&max<in[i]){
                max = in[i];
            }
        }
        return max==Double.MIN_VALUE?Double.NaN:max;
    }
    public static double nanMin(double[] in){
        double min = Double.MAX_VALUE;
        for(int i = 0; i<in.length;i++){
            if(!Double.isNaN(in[i])&&min>in[i]){
                min = in[i];
            }
        }
        return min==Double.MAX_VALUE?Double.NaN:min;
    }
    public static double[] normalize(double[] in, double max, double min){
        double diff = 0.5*(min + max);
        double div = 0.5*(max - min);
        return mult(add(in,-diff),1/div);
    }
    public static Tuple<double[],Double> randomlyHideValues(double[] in,double p){
        int count = 0;
        UniformRealDistribution dist = new UniformRealDistribution(0,1);
        for(int i  = 0; i<in.length;i++){
            if(dist.sample()>p) {
                in[i] = Double.NaN;
                count++;
            }
        }
        return new Tuple<>(in,1-(double)count/(double)in.length);

    }
    public static Tuple<double[],Double> randomlyHideConsecutiveEntries(double[] array,double pObservationRow,int longestStretch,int gap) {
        int n = array.length;
        int valuesToHide = (int)((1.0 - pObservationRow) * n);
        int count = 0;
        int countStart = 0;
        int i = 0;
        UniformRealDistribution dist = new UniformRealDistribution(0,1);
        while (i < n) {
            if (dist.sample() > pObservationRow) {
                countStart ++;
                int toHide = longestStretch;
                int startingIndex = i + (int)(dist.sample() * (gap - toHide));

                if (toHide + startingIndex > (i + gap)) {
                    toHide = (i + gap) - startingIndex;
                }
                for(int j = startingIndex;startingIndex<toHide;j++){
                    array[j] = Double.NaN;
                }
                count += toHide;
                valuesToHide -= toHide;
                if (valuesToHide <= 0){
                    break;
                }

            }
            i += gap;
        }

        double p_obs = (double)count/(double)n;
        return new Tuple<>(array,1-p_obs);

    }
    public static double rmse(double[] a,double[] b){
        double tot = 0;
        for(int i = 0; i<a.length;i++){
            tot+= Math.pow(a[i]-b[i],2);
        }
        return Math.sqrt(tot);
    }
    public static double nanMax(double[][] in){
        double max = nanMax(in[0]);
        for(int i = 1;i<in.length;i++){
            double temp;
            if(max<(temp=nanMax(in[i]))){
                max = temp;
            }
        }
        return max;
    }
    public static double nanMin(double[][] in){
        double min = nanMin(in[0]);
        for(int i = 1;i<in.length;i++){
            double temp;
            if(min>(temp=nanMin(in[i]))){
                min = temp;
            }
        }
        return min;
    }
    public static double[][] unifromRand(int n, int m){
        UniformRealDistribution x = new UniformRealDistribution(0,1);
        double[][] ret = new double[n][m];
        for(int i = 0; i<ret.length;i++){
            for(int j = 0; j<m;j++){
                ret[i][j] = x.sample();
            }
        }
        return ret;
    }
    public static double[][] unifromRand(int n, int m, int seed){
        UniformRealDistribution x = new UniformRealDistribution(0,1);
        x.reseedRandomGenerator(seed);
        double[][] ret = new double[n][m];
        for(int i = 0; i<ret.length;i++){
            for(int j = 0; j<m;j++){
                ret[i][j] = x.sample();
            }
        }
        return ret;
    }
    public static double[] unifromRand(int n){
        UniformRealDistribution x = new UniformRealDistribution(0,1);
        double[] ret = new double[n];
        for(int i = 0; i<ret.length;i++){
            ret[i] = x.sample();
        }
        return ret;
    }
    public static double mean(double[][] in){
        double tot = 0;
        int count=0;
        for(int i = 0; i<in.length;i++){
            for(int j = 0; j<in[i].length;j++){
                tot += in[i][j];
                count++;
            }
        }
        return count==0 ? 0 : tot/count;
    }
    public static double mean(double[] toArray) {
        double tot = 0;
        int count=0;
        for(int i = 0; i<toArray.length;i++){
                tot += toArray[i];
                count++;
        }
        return count==0 ? 0 : tot/count;
    }

    public static Double[][] toObject(double[][] values) {
        Double[][] ret = new Double[values.length][];
        for(int i = 0; i<values.length;i++){
            ret[i] = new Double[values[i].length];
            for(int j = 0; j<values[i].length;j++){
                ret[i][j] = values[i][j];
            }
        }
        return ret;
    }

    public static Double[] toObject(double[] values) {
        Double[] ret = new Double[values.length];
        for(int i = 0; i<values.length;i++){
                ret[i] = values[i];
        }
        return ret;
    }

    public static boolean isFinite(double[][] in) {
        for(int i = 0; i<in.length;i++){
            for(int j = 0; j<in[i].length;j++){
                if(!Double.isFinite(in[i][j]))
                    return false;
            }
        }
        return true;
    }
    public static double[] getInterval(double[] ar, int index,int length){
        double[] returner = new double[length];
        System.arraycopy(ar,index,returner,0,length);
        return returner;
    }

    public static double sum(double[] w, int index, int len) {
        double sum = 0;
        int count = 0;
        while (count<len){
            sum+= w[index+count++];
        }
        return sum;
    }
    public static double sum(double[] w) {
        return sum(w,0,w.length);
    }

    public static double[] multInverse(double[] arr) {
        double[] ret = new double[arr.length];
        int count = 0;
        for(double d : arr){
            ret[count++] = 1/d;
        }
        return ret;
    }

    public static double[] abs(double[] doubles) {
        double[] ret = new double[doubles.length];
        int count = 0;
        for(double d : doubles){
            ret[count++] = Math.abs(d);
        }
        return ret;
    }

    public static int max(int[] lags) {
        int max = Integer.MIN_VALUE;
        for(int l : lags){
            if(l>max)
                max=l;
        }
        return max;
    }
    public static double max(double[] in){
        double max = Double.MIN_VALUE;
        for(double d : in){
            max = d>max ? d : max;
        }
        return max;
    }

    public static double[][] create(int k, int l, double init) {
        double[][] ret = new double[k][l];
        for(int i = 0; i<k;i++){
            for (int j = 0; j<l;j++){
                ret[i][j] = init;
            }
        }
        return ret;
    }
    public static double[] create(int k, double init) {
        double[] ret = new double[k];
        for(int i = 0; i<k;i++){
                ret[i] = init;
        }
        return ret;
    }

    public static double[] repeat(double[] arr, int repeats) {
        double[] ret = new double[arr.length*repeats];
        for(int i = 0; i<repeats;i++){
            System.arraycopy(arr,0,ret,i*arr.length,arr.length);
        }
        return ret;
    }
    //insert Element e into Array r at index i ([1,2,3],5,1) -> [1,5,2,4]
    // does not change input Array
    public static double[] insert(double[] r,double e,int i ) {
        double[] ret = new double[r.length+1];
        int index = 0;
        for (int j = 0; j<ret.length;j++,index++){
            j = j == i ? ++j : j;
            ret[j] = r[index];
        }
        ret[i] = e;
        return ret;
    }



}
