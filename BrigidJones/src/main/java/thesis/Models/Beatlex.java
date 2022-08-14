package thesis.Models;
import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.linear.MatrixUtils;
import thesis.TSModel;
import thesis.Tools.TimeSeries;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

//todo: test functionality for multidimensional TS
//test failed need fixing in higher dimensional case
public class Beatlex extends TSModel {
    private static final double CF = 8;

    public String getParams() {
        return "Smin: "+Smin+"\nSmax: "+Smax+"\nmax_dist: "+max_dist;
    }


    private boolean autoTune;

    //hyper parameter
    TimeSeries X;
    int Smin;
    int Smax;
    int max_dist;
    double verbose;

    //defaultparams

    private static final  int defaultSmin = 10;
    private static final int defaultSmax = 350;
    private static  final int defaultMaxDist = 250;
    private static final double defaultVerbose = 0;

    public Beatlex(){
        super(new HashMap<>());
    }
    public Beatlex(HashMap<String, ArrayList<String>> in) {
        super(in);
    }


    protected void parse(HashMap<String, ArrayList<String>> input) {
        Smax = input.containsKey("-smax") ? new Double(input.get("-smax").get(0)).intValue() : defaultSmax;
        Smin = input.containsKey("-smin") ? new Double(input.get("-smin").get(0)).intValue() : defaultSmin;

        max_dist = input.containsKey("-maxdist") ? new Double(input.get("-maxdist").get(0)).intValue() : defaultMaxDist;


        verbose = input.containsKey("-verbose") ? new Double(input.get("-verbose").get(0)) : defaultVerbose;

    }


    public Beatlex(int smin, int smax, int max_dist, double verbose) {
        Smax = smax;
        Smin = smin;
        this.max_dist = max_dist;
        this.verbose = verbose;
        autoTune = false;

    }

    protected MultiOutput compute_all_horizons(MultiOutput outi){
        MatlabEngine eng;
        Object[] out;
        try {
            eng = MatlabEngine.startMatlab();
            if(autoTune) {
                out = eng.feval(3,"tune_hyperparams",(Object)outi.getTrain().getData());
                Smin = (int)((double)out[0]);
                Smax = (int)((double)out[1]);
                max_dist = (int)((double)out[2]);
            }

            out = eng.feval(9,"forecast_seq",outi.getTrain().getData(),outi.getTest().getColumnDimension(),Smin,Smax,max_dist,verbose);
            outi.setRunTime((double)out[1]);
            for(int i = 1; i<=outi.getTest().getColumnDimension();i++) {
                outi.add_horizon(i,ts.getRowDimension() == 1 ? MatrixUtils.createRowRealMatrix((double[]) out[0]) : MatrixUtils.createRealMatrix((double[][]) out[0]));
            }


        } catch (EngineException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return outi;

    }

    @Override
    public Output evaluate() {
        MatlabEngine eng;
        Object[] out;
        try {
            eng = MatlabEngine.startMatlab();
            if(autoTune) {
                out = eng.feval(3,"tune_hyperparams",(Object)o.getTrain().getData());
                Smin = (int)((double)out[0]);
                Smax = (int)((double)out[1]);
                max_dist = (int)((double)out[2]);
            }

            out = eng.feval(9,"forecast_seq",o.getTrain().getData(),o.getTest().getColumnDimension(),Smin,Smax,max_dist,verbose);
            o.setRunTime((double)out[1]);
            o.setForecast(ts.getRowDimension() == 1 ? MatrixUtils.createRowRealMatrix((double[])out[0]) : MatrixUtils.createRealMatrix((double[][])out[0]));


        } catch (EngineException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return o;
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
    public String toString(){
        return "beatlex";
    }

    @Override
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }
}
