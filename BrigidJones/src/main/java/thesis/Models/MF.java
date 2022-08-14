package thesis.Models;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import thesis.TSModel;
import thesis.Tools.TimeSeries;
import thesis.Tools.Timer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;
//todo: predict method needs discussion

public class MF extends TSModel {

    //Model variables
    RealVector weig;
    RealMatrix U;
    RealVector V;
    RealMatrix BLK;


    //all optimizers share theese fields
    double dim;
    int nord;
    double reg;

    Optimizer opt;

    //FP&FT specific
    double pen_v;
    double pen_u;

    //FT specific
    double tol_u;

    private RealMatrix PRED;
    private RealMatrix train;


    //default params
     private static final Optimizer defualtopt = Optimizer.FP;
     private static final double defaultpen_u = 0.1;
     private static final double defaultpen_v = 1e-4;
     private static final double defaulttolu = 0.05;
     private static final double defaultreg = 1;

     private static final double defaultdim = 0.02;
     private static final int defaultnord = 24;

    protected void parse(HashMap<String, ArrayList<String>> input) {
        if(input.containsKey("-opti")){
            String temp = input.get("-opti").get(0);
            if(temp.equalsIgnoreCase("zt"))
                opt = Optimizer.ZT;
            else if(temp.equalsIgnoreCase("fp"))
                opt = Optimizer.FP;
            else
                opt = Optimizer.FT;
        }
        else
            opt = defualtopt;

        pen_u = input.containsKey("-penu") ? new Double(input.get("-penu").get(0)) : defaultpen_u;
        pen_v = input.containsKey("-penv") ? new Double(input.get("-penv").get(0)) : defaultpen_v;
        reg = input.containsKey("-reg") ? new Double(input.get("-reg").get(0)) : defaultreg;
        tol_u = input.containsKey("-tolu") ? new Double(input.get("-tolu").get(0)) : defaulttolu;

        dim = input.containsKey("-dim") ? new Double(input.get("-dim").get(0)) : defaultdim;
        nord = input.containsKey("-nord") ? new Integer(input.get("-nord").get(0)) : defaultnord;
    }











    @Override
    protected Output evaluate() {
        MatlabEngine eng;
        dim = dim<1 ? (int)(ts.getRowDimension()*dim) : (int)dim;
        train = o.getTrain();
        try {
            Object[] out;
            eng = MatlabEngine.startMatlab();
            if(opt == Optimizer.FP) {
                out = eng.feval(6, "filter_FP", o.getTrain().getData(),createIdMat(o.getTrain()).getData(),(double)dim,pen_u,pen_v,(double)nord,reg);
            }
            else if(opt == Optimizer.FT){
                out = eng.feval(6,"filter_FT",o.getTrain().getData(),createIdMat(o.getTrain()).getData(),(double)dim,tol_u,pen_v,(double)nord,reg);
            }
            else{
                out = eng.feval(6,"filter_ZT",o.getTrain().getData(),createIdMat(o.getTrain()).getData(),(double)dim,pen_v,(double)nord,reg);
            }
            double learnTime = (double)out[3];

            V = (dim == 1) ? MatrixUtils.createRealVector(new double[]{(double)out[1]}) : MatrixUtils.createRealVector((double[])out[1]);


            if(ts.getRowDimension()==1)
                U = MatrixUtils.createRealMatrix(new double[][]{{(double)out[0]}});
            else
                U = dim == 1 ? MatrixUtils.createRowRealMatrix((double[])out[0]) : MatrixUtils.createRealMatrix((double[][])out[0]);


            if(nord == 1&&dim==1)
                BLK = MatrixUtils.createRealMatrix(new double[][]{{(double)out[4]}});
            else if(nord == 1)
                BLK = MatrixUtils.createColumnRealMatrix((double[])out[4]);
            else if(dim == 1)
                BLK = MatrixUtils.createRowRealMatrix((double[])out[4]);
            else
                BLK = MatrixUtils.createRealMatrix((double[][])out[4]);

            PRED = MatrixUtils.createRealMatrix((double[][])out[5]);

            weig = nord == 1 ? MatrixUtils.createRealVector(new double[]{(double)out[2]}) : MatrixUtils.createRealVector((double[])out[2]);
            Timer t = new Timer();
            t.start();
            o.setForecast(predict(o.getTest()).toMatrix());
            o.setRunTime(t.stop()+learnTime);



        } catch (EngineException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return o;

    }
    protected MultiOutput compute_all_horizons(MultiOutput outi){

        MatlabEngine eng;
        dim = dim<1 ? (int)(ts.getRowDimension()*dim) : (int)dim;
        train = outi.getTrain();
        try {
            Object[] out;
            eng = MatlabEngine.startMatlab();
            if(opt == Optimizer.FP) {
                out = eng.feval(6, "filter_FP", outi.getTrain().getData(),createIdMat(outi.getTrain()).getData(),(double)dim,pen_u,pen_v,(double)nord,reg);
            }
            else if(opt == Optimizer.FT){
                out = eng.feval(6,"filter_FT",outi.getTrain().getData(),createIdMat(outi.getTrain()).getData(),(double)dim,tol_u,pen_v,(double)nord,reg);
            }
            else{
                out = eng.feval(6,"filter_ZT",outi.getTrain().getData(),createIdMat(outi.getTrain()).getData(),(double)dim,pen_v,(double)nord,reg);
            }
            double learnTime = (double)out[3];

            V = (dim == 1) ? MatrixUtils.createRealVector(new double[]{(double)out[1]}) : MatrixUtils.createRealVector((double[])out[1]);


            if(ts.getRowDimension()==1)
                U = MatrixUtils.createRealMatrix(new double[][]{{(double)out[0]}});
            else
                U = dim == 1 ? MatrixUtils.createRowRealMatrix((double[])out[0]) : MatrixUtils.createRealMatrix((double[][])out[0]);


            if(nord == 1&&dim==1)
                BLK = MatrixUtils.createRealMatrix(new double[][]{{(double)out[4]}});
            else if(nord == 1)
                BLK = MatrixUtils.createColumnRealMatrix((double[])out[4]);
            else if(dim == 1)
                BLK = MatrixUtils.createRowRealMatrix((double[])out[4]);
            else
                BLK = MatrixUtils.createRealMatrix((double[][])out[4]);

            PRED = MatrixUtils.createRealMatrix((double[][])out[5]);

            weig = nord == 1 ? MatrixUtils.createRealVector(new double[]{(double)out[2]}) : MatrixUtils.createRealVector((double[])out[2]);
            Integer OG_windowSize = stepSize;
            for(stepSize = 1 ; stepSize<=outi.getTest().getColumnDimension();stepSize++) {
                outi.add_horizon(stepSize,predict(outi.getTest()).toMatrix());
            }
            stepSize = OG_windowSize;
            outi.setRunTime(learnTime);



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
    public boolean is1D() {
        return false;
    }

    @Override
    public boolean isND() {
        return true;
    }

    @Override
    public boolean missingValuesAllowed() {
        return true;
    }

    @Override
    public String toString(){
        return "mf";
    }

    public static enum Optimizer{
        FP,
        FT,
        ZT;
    }

    public MF(Optimizer opt, double pen_u, double pen_v, int dim, int nord, double reg, double tol_u){
        this.opt = opt;
        this.pen_u = pen_u;
        this.pen_v = pen_v;
        this.dim = dim;
        this.nord = nord;
        this.reg = reg;
        this.tol_u = tol_u;
    }
    public MF(){
        super(new HashMap<>());
    }
    public MF(HashMap<String, ArrayList<String>> in) {
        super(in);
    }

    public HashMap<String,String[]> getSearchSpace(){
        HashMap<String,String[]> space = new HashMap<>();
        space.put("-penu",new String[]{"10","1","0.1"});
        space.put("-reg",new String[]{"0.5","1","5"});
        space.put("-dim",new String[]{"0.01","0.02","0.05","0.1","0.2"});
        space.put("-nord",new String[]{"10","15","25"});
        return space;
    }



    public TimeSeries predict(RealMatrix test_set){
        int predictionLength = test_set.getColumnDimension();
        int count = 0;
        double[][] prediction = new double[predictionLength][];
        SingularValueDecomposition svd = new SingularValueDecomposition(U.transpose());
        RealMatrix uTpseudo = svd.getSolver().getInverse();
        RealMatrix BLK_TEMP = BLK.copy();
        RealVector v_hat;
        RealVector x_hat;
        for(int i = 0; i<predictionLength;i++){
            if(i%stepSize==0&&i!=0){
                for(int j = 0; j<stepSize&&j<BLK_TEMP.getColumnDimension();j++){
                    BLK_TEMP.setColumn(j,uTpseudo.operate(test_set.getColumnVector(i-stepSize+j)).toArray());

                }
            }
            v_hat = BLK_TEMP.operate(weig);
            x_hat = U.transpose().operate(v_hat);
            prediction[i] =x_hat.toArray();
            BLK_TEMP.setSubMatrix(BLK.getSubMatrix(0,BLK_TEMP.getRowDimension()-1,0,BLK_TEMP.getColumnDimension()>1 ? BLK_TEMP.getColumnDimension()-2 : 0).getData(),0,BLK_TEMP.getColumnDimension()>1 ? 1 : 0);
            BLK_TEMP.setColumnVector(0,v_hat);


        }
        return new TimeSeries(prediction);

    }

    public Output trainPred(){
        Output o = new Output();
        o.setForecast(PRED);
        o.setTest(train);
        o.setTrain(MatrixUtils.createRealMatrix(PRED.getRowDimension(),1));
        return o;

    }

//testScript




   static RealMatrix createIdMat(RealMatrix in) {
            ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
            ArrayList<Integer> counter;
        for(int i = 0; i<in.getColumnDimension();i++){
            counter = new ArrayList<>();
            for(int j = 0; j<in.getRowDimension();j++){
                if(Double.isFinite(in.getEntry(j,i))){
                    counter.add(j);
                }
            }
            ret.add(counter);
        }
        RealMatrix reti = MatrixUtils.createRealMatrix(in.getRowDimension(),in.getColumnDimension());
        for(int i = 0; i<ret.size();i++){
            for(int j = 0; j<ret.get(i).size();j++){
                reti.setEntry(j,i,ret.get(i).get(j)+1);
            }
        }
        return reti;
    }

}
