package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import thesis.TSModel;
import thesis.Tools.*;

import java.io.IOException;
import java.util.*;

public class BeatlexAlt extends TSModel {
    private static final double alpha = 0.2;
    private static final double new_cluster_threshold = 0.3;
    private static final int max_order = 3;

    private static final  int defaultSmin = 10;
    private static final int defaultSmax = 350;
    private static  final int defaultMaxDist = 250;

    int smin;
    int smax;
    int max_dist;
    private static final int kmax = 5;

    int cut_of_index = -1;

    ArrayList<RealMatrix> vocabulary;
    ArrayList<Integer> prefixes;
    ArrayList<Integer> map;

    public BeatlexAlt(){
        super();
    }
    public BeatlexAlt(HashMap<String,ArrayList<String>> in){
        super(in);
    }

//tode: This will need testing
    private int greedy_best_length(RealMatrix remainder){
        //edge case no search possible
        if(smin>=remainder.getColumnDimension())
            return remainder.getColumnDimension();

        ArrayList<Double> errors = new ArrayList<>();

        for(int i = smin-1;i<smax;i++){
            //check if end of time series is hit
            if(i>=remainder.getColumnDimension()-1)
                break;

            RealMatrix Xtemp = remainder.getSubMatrix(0,remainder.getRowDimension()-1,i+1,remainder.getColumnDimension()-1);
            RealMatrix voc_temp = remainder.getSubMatrix(0,remainder.getRowDimension()-1,0,i);
            vocabulary.add(voc_temp);
            Triple<Integer,Integer,Double> res = best_prefix(Xtemp);
            errors.add(res.getVal3());
            vocabulary.remove(voc_temp);
        }
        int best_length = smin+errors.indexOf(Collections.min(errors));
        return best_length;

    }

    private RealMatrix merge(RealMatrix vocab, RealMatrix interval){
        DTW dtw = new DTW(vocab,interval,max_dist,true);
        RealMatrix ret = MatrixUtils.createRealMatrix(vocab.getRowDimension(),vocab.getColumnDimension());
        ArrayList<Tuple<Integer,Integer>> path = new ArrayList<>();
        for(int i = dtw.path.size()-1;i>=0;i--){
            path.add(dtw.path.get(i));
        }
        int j;
        for(int i = 0; i<path.size();i+=j){
            int vocab_index = path.get(i).getVal1();

            j = 0;
            ArrayList<RealVector> interval_vecs = new ArrayList<>();
            while (i+j<path.size()&&path.get(i+j).getVal1() == vocab_index){
                interval_vecs.add(interval.getColumnVector(path.get(i+j).getVal2()));
                j++;
            }
            RealVector sum = MatrixUtils.createRealVector(new double[vocab.getRowDimension()]);
            for(RealVector v : interval_vecs){
                sum = sum.add(v);
            }
            sum.mapDivideToSelf(interval_vecs.size());

            ret.setColumnVector(vocab_index,vocab.getColumnVector(vocab_index).mapMultiply(1-alpha).add(sum.mapMultiply(alpha)));
        }
        return ret;
    }
//have to work over scenario where remaining matrix is shorter than smax
    private void summurize_seq(RealMatrix X){
        map = new ArrayList<>();
        vocabulary = new ArrayList<>();
        prefixes = new ArrayList<>();

        int best_length = greedy_best_length(X);
        int best_vocab;
        double best_error;
        int cur_idx = 0;

        //conpute mean_dev
        double mean_dev = new Variance().evaluate(MatrixTools.flatten(X));


        prefixes.add(cur_idx);
        vocabulary.add(X.getSubMatrix(0,X.getRowDimension()-1,0,best_length-1));
        map.add(0);
        if(best_length>=X.getColumnDimension()){
            return;
        }


        RealMatrix Xcur = X.getSubMatrix(0,X.getRowDimension()-1,best_length,X.getColumnDimension()-1);
        cur_idx = best_length;
        while (true){
            Triple<Integer,Integer,Double> ret = best_prefix(Xcur);
            best_length = ret.getVal2();
            best_vocab = ret.getVal1();
            best_error = ret.getVal3();
            //check if not the remaining time series can be better matched to a subsequence of a vocabulary term
            if(Xcur.getColumnDimension()<=smax){
                Triple<Integer,Integer,Double> result = best_prefix_vocab(Xcur);
                if(result.getVal3()<=best_error||Xcur.getColumnDimension()<=smin){
                    map.add(result.getVal1());
                    cut_of_index = result.getVal2();
                    break;

                }
            }

            if(best_error <= new_cluster_threshold * mean_dev || vocabulary.size() >= kmax){
                RealMatrix newVocab = merge(vocabulary.get(best_vocab),Xcur.getSubMatrix(0,Xcur.getRowDimension()-1,0,best_length-1));
                vocabulary.set(best_vocab,newVocab);

            }
            else{
                best_length = greedy_best_length(Xcur);
                vocabulary.add(Xcur.getSubMatrix(0,Xcur.getRowDimension()-1,0,best_length-1));
                best_vocab = vocabulary.size()-1;

            }
            prefixes.add(cur_idx);
            cur_idx += best_length;
            map.add(best_vocab);
            if (best_length>=Xcur.getColumnDimension())
                break;

            Xcur = Xcur.getSubMatrix(0,Xcur.getRowDimension()-1,best_length,Xcur.getColumnDimension()-1);

        }
    }

    private RealMatrix forecast(int length){
        //learn markov model up to max_lenght chain length
        ArrayList<HashMap<String,Integer>> markvov_counts = new ArrayList<>();

        //initilize markov models;
        for(int i = 0; i<=max_order;i++) {
            markvov_counts.add(new HashMap<>());
        }
        //count number of occurences of given pattern.
        for(int j = 0; j<markvov_counts.size();j++) {
            for (int i = j; i < map.size();i++){
                HashMap<String,Integer> cur_markov = markvov_counts.get(j);
                List<Integer> cur_seq =  map.subList(i-j,i+1);
                String cur_code = encode(cur_seq);
                if(cur_markov.containsKey(cur_code)){
                    cur_markov.put(cur_code,cur_markov.get(cur_code)+1);
                }
                else{
                    cur_markov.put(cur_code,1);
                }
            }
        }
        //associate each sequence with the most probable following term
        ArrayList<HashMap<String,Integer>> markov_maps = new ArrayList<>();
        ArrayList<HashMap<String,Integer>> markov_maxs = new ArrayList<>();
        for(int i = 0; i<markvov_counts.size();i++){
            markov_maps.add(new HashMap<>());
            markov_maxs.add(new HashMap<>());
            for(String key : markvov_counts.get(i).keySet()){
                String newKey = key.substring(0,key.length()-1);
                Integer term = new Integer(Character.toString(key.charAt(key.length()-1)));
                if(!markov_maps.get(i).containsKey(newKey)){
                    markov_maps.get(i).put(newKey,term);
                    markov_maxs.get(i).put(newKey,markvov_counts.get(i).get(key));
                }
                else{
                    if(markov_maxs.get(i).get(newKey)<markvov_counts.get(i).get(key)){
                        markov_maxs.get(i).put(newKey,markvov_counts.get(i).get(key));
                        markov_maps.get(i).put(newKey,term);
                    }
                }
            }
        }

        int cur_length = 0;
        RealMatrix forecast = MatrixUtils.createRealMatrix(ts.getRowDimension(),length);
        List<Integer> past = map.subList(map.size() - markov_maps.size()+1, map.size());

        //first finish the maybe unfinished previous pattern:
        RealMatrix last_vocab = vocabulary.get(map.get(map.size() - 1));
        if(cut_of_index!=-1&&cut_of_index<last_vocab.getColumnDimension() -1) {
            RealMatrix tempi = last_vocab.getSubMatrix(0, last_vocab.getRowDimension() - 1, cut_of_index + 1, last_vocab.getColumnDimension() - 1);
            if(tempi.getColumnDimension()>forecast.getColumnDimension())
                tempi = tempi.getSubMatrix(0,tempi.getRowDimension()-1,0,forecast.getColumnDimension()-1);

            forecast.setSubMatrix(tempi.getData(), 0, 0);
            cur_length = last_vocab.getColumnDimension() - (cut_of_index + 1);
        }

        //forecast sequence
        while (cur_length<length) {
            ArrayList<Integer> temp = new ArrayList<>(past);
            int next = -1;
            while (true) {
                if (markov_maps.get(temp.size()).containsKey(encode(temp))) {
                    next = markov_maps.get(temp.size()).get(encode(temp));
                    break;
                }
                temp.remove(0);
            }
            int cut_of = Math.min(vocabulary.get(next).getColumnDimension()-1,length-cur_length-1);
            forecast.setSubMatrix(vocabulary.get(next).getSubMatrix(0,ts.getRowDimension()-1,0,cut_of).getData(),0,cur_length);
            cur_length += vocabulary.get(next).getColumnDimension();
            past.remove(0);
            past.add(next);
        }


        return forecast;




    }

    private String encode(List<Integer> cur_seq) {
        String ret = "";
        for(Integer i : cur_seq){
            ret += i.toString();
        }
        return ret;
    }

    //
private int allign(RealMatrix vocab, RealMatrix segment){
        DTW dtw = new DTW(vocab,segment,max_dist,true);
        RealVector errors = dtw.D.getColumnVector(segment.getColumnDimension()-1);
        ArrayList<Double> temp = new ArrayList<>();
        for(int i = 0; i<errors.getDimension();i++){
            temp.add(errors.getEntry(i));
        }
        return temp.indexOf(Collections.min(temp));
}
    //tuple with best vocabulary term and prefix length and its corresponding error.
    private Triple<Integer,Integer,Double> best_prefix(RealMatrix remainder){
        int best_vocab;
        int global_best_length;


        int length = Math.min(remainder.getColumnDimension(),smax);
        RealMatrix curX = remainder.getSubMatrix(0,remainder.getRowDimension()-1,0,length-1);

        ArrayList<Integer> vocabs_best_length = new ArrayList<>();
        ArrayList<Double> vocabs_best_error = new ArrayList<>();
        for(int i = 0; i<vocabulary.size();i++) {
            DTW dtw = new DTW(curX,vocabulary.get(i),max_dist,true);
            RealVector errors = dtw.D.getColumnVector(vocabulary.get(i).getColumnDimension()-1);

            //extract DTW errors for significant region
            ArrayList<Double> values = new ArrayList<>();
            double best_length_error = 0;
            int best_length = 0;
            if(length<=smin){
                best_length_error = errors.getEntry(length-1)/length;
                best_length = length;
            }
            else {
                for (int j = smin-1; j < length; j++) {
                    values.add(errors.getEntry(j)/(j+1));
                }
                best_length_error = Collections.min(values);
                best_length = smin+ values.indexOf(best_length_error);
            }
            vocabs_best_length.add(best_length);
            vocabs_best_error.add(best_length_error);

        }
        best_vocab = vocabs_best_error.indexOf(Collections.min(vocabs_best_error));
        global_best_length = vocabs_best_length.get(best_vocab);

        return new Triple<>(best_vocab,global_best_length,vocabs_best_error.get(best_vocab));

    }
    //tuple with best vocabulary term and point at which the pattern is cut off and its corresponding error.
    private Triple<Integer,Integer,Double> best_prefix_vocab(RealMatrix remainder){
        int best_vocab;
        int global_best_index;



        RealMatrix curX = remainder;

        ArrayList<Integer> vocabs_best_length = new ArrayList<>();
        ArrayList<Double> vocabs_best_error = new ArrayList<>();
        for(int i = 0; i<vocabulary.size();i++) {
            DTW dtw = new DTW(vocabulary.get(i),curX,max_dist,true);
            RealVector errors = dtw.D.getColumnVector(curX.getColumnDimension()-1);

            //extract DTW errors for significant region
            ArrayList<Double> values = new ArrayList<>();
            double best_index_error = 0;
            int best_index = 0;
            for (int j = 0; j < errors.getDimension(); j++) {
                values.add(errors.getEntry(j)/(j+1));
            }
            best_index_error = Collections.min(values);
            best_index = values.indexOf(best_index_error);

            vocabs_best_length.add(best_index);
            vocabs_best_error.add(best_index_error);

        }
        best_vocab = vocabs_best_error.indexOf(Collections.min(vocabs_best_error));
        global_best_index = vocabs_best_length.get(best_vocab);

        return new Triple<>(best_vocab,global_best_index,vocabs_best_error.get(best_vocab));

    }
    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        smax = in.containsKey("-smax") ? new Double(in.get("-smax").get(0)).intValue() : defaultSmax;
        smin = in.containsKey("-smin") ? new Double(in.get("-smin").get(0)).intValue() : defaultSmin;

        max_dist = in.containsKey("-maxdist") ? new Double(in.get("-maxdist").get(0)).intValue() : defaultMaxDist;

    }

    @Override
    protected Output evaluate() {
        summurize_seq(o.getTrain());
        o.setForecast(forecast(o.getTest().getColumnDimension()));
        return o;
    }


    private static class DTW{
        double error;
        ArrayList<Tuple<Integer,Integer>> path;
        RealMatrix D;
        double k;

        public DTW(RealMatrix t, RealMatrix r, int max_dist,boolean penalty){
            int N = t.getColumnDimension();
            int M = r.getColumnDimension();
            D = MatrixUtils.createRealMatrix(N,M);
            RealMatrix d = MatrixUtils.createRealMatrix(N,M);
            MatrixTools.replace(D,0,Double.MAX_VALUE);
            for(int n = 0; n<N;n++){
                for(int m = 0; m<M;m++){
                    d.setEntry(n,m,t.getColumnVector(n).getDistance(r.getColumnVector(m)));
                }
            }
            //initialize error matrix with edge cases
            D.setEntry(0,0,d.getEntry(0,0));

            for(int n = 1; n<N;n++){
                D.setEntry(n,0,D.getEntry(n-1,0)+d.getEntry(n,0));
            }
            for(int m = 1; m<M;m++){
                D.setEntry(0,m,D.getEntry(0,m-1)+d.getEntry(0,m));
            }
            //encoding cost of multimapp
            double encode_cost = 1;
            double mcost = encode_cost * Math.sqrt(new Variance().evaluate(MatrixTools.flatten(t))) * (Math.log(M)/Math.log(2));
            double ncost = encode_cost * Math.sqrt(new Variance().evaluate(MatrixTools.flatten(r))) * (Math.log(N)/Math.log(2));
            if (!penalty){
                mcost = 0;
                ncost = 0;
            }
            //fill errors recurivly
            for(int n = 1; n<N;n++){
                int m_min = Math.max(1, n-max_dist);
                int m_max = Math.min(M, n+max_dist);
                for(int m = m_min; m<m_max;m++){
                    D.setEntry(n,m,d.getEntry(n,m)+Math.min(Math.min(D.getEntry(n-1,m)+mcost, D.getEntry(n-1,m-1)),D.getEntry(n,m-1)+ncost));
                }
            }
            error = D.getEntry(N-1,M-1);
            //Find the best Path
            path = new ArrayList<>();
            int n = N-1;
            int m = M-1;
            path.add(new Tuple<>(n,m));
            k=1;
            while(n!=0 || m != 0){
                if(n == 0)
                    m--;
                else if(m == 0)
                    n--;
                else{
                    double min = Math.min(Math.min(D.getEntry(n-1,m), D.getEntry(n-1,m-1)),D.getEntry(n,m-1));
                    if(D.getEntry(n-1,m)==min)
                        n--;
                    else if(D.getEntry(n-1,m-1)==min){
                        n--;
                        m--;
                    }
                    else
                        m--;
                }
                k++;
                path.add(new Tuple<>(n,m));

            }
        }

    }

    @Override
    public boolean is1D() {
        return true;
    }

    @Override
    public boolean isND() {
        return true;
    }

    @Override
    public boolean missingValuesAllowed() {
        return false;
    }

    @Override
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }


}
