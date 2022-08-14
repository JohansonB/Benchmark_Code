package thesis.Models;

//Only use for 1-D Timeseries

import com.mathworks.matlab.types.Struct;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import thesis.TSModel;
import thesis.Tools.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

import com.mathworks.engine.*;

public class ARIMA_Online extends TSModel {


    public enum Optimization_Algorithm{
        OGD,
        ONS;
    }
    private double[] RMSE;

    //defaultParams

    private static final int defaultMK = 10;
    private static final int defaultDif = 0;
    private static final double defaultLrate = 0.0001;
    private static final double defaultEpsilon = 0.0001;
    private static final Optimization_Algorithm defaultOption = Optimization_Algorithm.OGD;


    private double lrate;
    private int mk;
    private RealVector w;
    private double epsilon;
    private int dif;
    private Optimization_Algorithm option;

    public ARIMA_Online(){
        super(new HashMap<>());
    }
    public ARIMA_Online(HashMap<String, ArrayList<String>> in) {
        super(in);
    }


    public ARIMA_Online(double lrate, int mk, int dif, double epsilon, Optimization_Algorithm op ) {
        this.mk  = mk;
        this.dif = dif;
        this.lrate  = lrate;
        w = MatrixTools.randomVector(mk,-1,1);
        this.option = op;
        this.epsilon = epsilon;
    }
    public String getParams(){
        return "dif : "+dif+" epsilon: "+epsilon+" mk: "+mk;
    }


    protected MultiOutput compute_all_horizons(MultiOutput out){
        MatlabEngine eng;
        try {
            eng = MatlabEngine.startMatlab();


            Tuple<TimeSeries,double[][]> tempi = new TimeSeries(ts).differenceAndLastOrders(dif);
            TimeSeries test = tempi.getVal1().getSubSeries(tempi.getVal1().length()-out.getTest().getColumnDimension(),out.getTest().getColumnDimension());
            TimeSeries train = tempi.getVal1().getSubSeries(0,tempi.getVal1().length()-out.getTest().getColumnDimension());

            String s = option == Optimization_Algorithm.OGD ? "arima_ogd" : "arima_ons";
            System.out.println(s + " "+this);
            Object[] results = eng.feval(3,s, tempi.getVal1().toMatrix().getRowVector(0).toArray(), new Struct("mk",(double)mk,"lrate",lrate,"init_w",w.toArray(),"t_tick",1,"epsilon",epsilon));
            RMSE =  (double[]) results[0];
            w = MatrixUtils.createRealVector((double[])results[1]);
            double learnTime = (double)results[2];
            Integer OG_stepsize = stepSize;
            for(stepSize = 1;stepSize<=test.length();stepSize++) {
                RealVector pred = predict(test.toVector(), train.toVector().getSubVector(train.length() - mk, mk));
                out.add_horizon(stepSize, new TimeSeries(pred).inverseDiff(tempi.getVal2(), true).toMatrix());
            }
            stepSize = OG_stepsize;
            out.setRunTime(learnTime);
            //if the execution passes the fit call ensures that ts has dimensionality 1xts.getColumnDimension()

        } catch (EngineException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return out;

    }

    //one step lookahead
    @Override
    protected Output evaluate(){
        MatlabEngine eng;
        try {
            eng = MatlabEngine.startMatlab();


            Tuple<TimeSeries,double[][]> tempi = new TimeSeries(ts).differenceAndLastOrders(dif);
            TimeSeries test = tempi.getVal1().getSubSeries(tempi.getVal1().length()-o.getTest().getColumnDimension(),o.getTest().getColumnDimension());
            TimeSeries train = tempi.getVal1().getSubSeries(0,tempi.getVal1().length()-o.getTest().getColumnDimension());

            String s = option == Optimization_Algorithm.OGD ? "arima_ogd" : "arima_ons";
            System.out.println(s + " "+this);
            Object[] results = eng.feval(3,s, tempi.getVal1().toMatrix().getRowVector(0).toArray(), new Struct("mk",(double)mk,"lrate",lrate,"init_w",w.toArray(),"t_tick",1,"epsilon",epsilon));
            RMSE =  (double[]) results[0];
            w = MatrixUtils.createRealVector((double[])results[1]);
            double learnTime = (double)results[2];
            Timer time = new Timer();
            time.start();
            RealVector pred = predict(test.toVector(),train.toVector().getSubVector(train.length()-mk,mk));
            o.setForecast(new TimeSeries(pred).inverseDiff(tempi.getVal2(),true).toMatrix());
            o.setRunTime(time.stop()+learnTime);
            //if the execution passes the fit call ensures that ts has dimensionality 1xts.getColumnDimension()

        } catch (EngineException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return o;
    }

    private RealVector predict(RealVector test,RealVector init) {
        RealVector ret = MatrixUtils.createRealVector(new double[test.getDimension()]);
        RealVector temp = MatrixUtils.createRealVector(new double[test.getDimension()+w.getDimension()]);
        temp.setSubVector(0,init);
        for(int i = 0; i<ret.getDimension();i++){
            if(i%stepSize==0&&i!=0){
                System.out.println(i);
                for(int j = 0; j<stepSize;j++){
                    temp.setEntry(i+w.getDimension()-stepSize+j,test.getEntry(i-stepSize+j));

                }
            }

            RealVector temptemp =  temp.getSubVector(i,w.getDimension());
            double pred = temptemp.dotProduct(w);
            ret.setEntry(i,pred);
            temp.setEntry(i+w.getDimension(),pred);

        }
        return ret;

    }

    @Override
    public boolean is1D() {
        return true;
    }

    @Override
    public boolean isND() {
        return false;
    }

    @Override
    public boolean missingValuesAllowed() {
        return false;
    }

    @Override
    public String toString(){
        return "oarima "+(option==Optimization_Algorithm.ONS ? "ONS" : "OGD");
    }

    public RealVector getW(){
        return w;
    }
    public RealVector getRMSE(){
        return MatrixUtils.createRealVector(RMSE);
    }

    public HashMap<String,String[]> getSearchSpace(){
        HashMap<String,String[]> space = new HashMap<>();
        space.put("-mk",new String[]{"5","10","15","20"});
        space.put("-dif",new String[]{"0","1","2"});
        space.put("-epsilon",new String[]{"0.0001","0.001","0.01","0.1","1"});
        return space;
    }

    protected void parse(HashMap<String, ArrayList<String>> input) {
        mk  = input.containsKey("-mk") ? new Integer(input.get("-mk").get(0)) : defaultMK;
        dif = input.containsKey("-dif") ? new Integer(input.get("-dif").get(0)) : defaultDif;
        lrate  = input.containsKey("-lrate") ? new Double(input.get("-lrate").get(0)) : defaultLrate;
        epsilon  = input.containsKey("-epsilon") ? new Double(input.get("-epsilon").get(0)) : defaultEpsilon;
        if(input.containsKey("-opt")){
            option = input.get("-opt").get(0).equalsIgnoreCase("ogd") ? Optimization_Algorithm.OGD : Optimization_Algorithm.ONS;
        }
        else {
            option=defaultOption;
        }
        w = MatrixTools.randomVector(mk,-1,1);
    }


    public static void main(String[] args) throws IOException{
        HashMap<String,String> h = new HashMap<>();
        h.put("-mk","100");

        ARIMA_Online m = new ARIMA_Online(GridSearcher.reformat(h));
        Vary_Result v = m.vary_parameter("Datasets\\solar_6_hourly_dataset_0_eval.csv",mult_range(10e-60,10e-20,10),"-lrate",h);
        v.store("C:\\Users\\41766\\Desktop\\Baechlor_Sourceroni_code\\H_T_Experiments\\o_arima\\solar\\vary_lrate_mk_100");
        Graph g = v.error_graph(new MAE());
        g.set_log_base_x(10);
        g.set_log_base_y(10);
        g.plot();
        v.plot_at(10);


    }
}
