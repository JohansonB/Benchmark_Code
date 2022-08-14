package thesis.Tools;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class CovarianceAnalysis {
    RealMatrix ts;
    double correlationThreshold;
    RealMatrix correlation;
    public CovarianceAnalysis(RealMatrix ts, double correlationThreshold){
        this.ts = ts;
        this.correlationThreshold = correlationThreshold;
        correlation = new PearsonsCorrelation().computeCorrelationMatrix(ts.transpose());
    }
    public CovarianceAnalysis(RealMatrix ts){
        this.ts = ts;
        this.correlationThreshold = 0.8;
        correlation = new PearsonsCorrelation().computeCorrelationMatrix(ts.transpose());
    }

    public ArrayList<ArrayList<Integer>> correlatedSeries(){
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        ArrayList<Integer> temp;
        for(int i = 0; i<correlation.getRowDimension();i++){
            temp = new ArrayList<>();
            for(int j = 0; j<correlation.getColumnDimension();j++){
                if(Math.abs(correlation.getEntry(i,j))>=correlationThreshold)
                    temp.add(j);
            }
            ret.add(temp);
        }
        return ret;

    }
    public ArrayList<ArrayList<Integer>> mergedCorrelatedSeries(){
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        ArrayList<Integer> temp = null;
        for(ArrayList<Integer> cor : correlatedSeries()){
            if((temp = contains(ret,cor))!=null){
                merge(temp,cor);
            }
            else{
                ret.add(cor);
            }
        }
        Collections.sort(ret, (Comparator<ArrayList>) (o1, o2) -> new Integer(o2.size()).compareTo(o1.size()));
        return ret;
    }
    public ArrayList<Double> min_corr(ArrayList<ArrayList<Integer>> cor_series){
        ArrayList<Double> ret = new ArrayList<>();
        for(ArrayList<Integer> block : cor_series){
            double min_cor = 1;
            int i1 = 0;
            int i2 = 0;
            for (Integer index : block){
                for(Integer index2 : block){
                    if(Math.abs(correlation.getEntry(index,index2))<=Math.abs(min_cor)){
                        min_cor = correlation.getEntry(index,index2);
                        i1 = index;
                        i2 = index2;
                    }
                }
            }
            System.out.println(i1);
            System.out.println(i2);
            ret.add(min_cor);
        }
        return ret;
    }

    private void merge(ArrayList<Integer> temp, ArrayList<Integer> cor) {
        for(Integer i : cor){
            if(!temp.contains(i)){
                temp.add(i);
            }
        }
    }

    private ArrayList<Integer> contains(ArrayList<ArrayList<Integer>> ret, ArrayList<Integer> cor) {
        for(Integer i : cor){
            for(ArrayList<Integer> cur : ret){
                if(cur.contains(i)){
                    return cur;
                }
            }
        }
        return null;
    }

    private void plot_class(ArrayList<ArrayList<Integer>> list, int index){
        ArrayList<Integer> indxs = list.get(index);
        TimeSeries time = new TimeSeries(ts);
        for(int i : indxs){
            time.get1DSubSeries(i).plot();
        }
    }

}
