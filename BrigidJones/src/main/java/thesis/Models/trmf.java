package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import thesis.TSModel;
import thesis.Tools.Graph;
import thesis.Tools.GridSearcher;
import thesis.Tools.MatrixTools;
import thesis.Tools.TimeSeries;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class trmf extends TSModel {
    private String dataPath = "Scripts\\C\\Temp\\TRMF_in.csv";
    private String testAccuracyScriptPath = "Scripts\\C\\TRMF\\trmf.exe";
    private String allHorizonScriptPath = "Scripts\\C\\TRMF\\TRMF_Model.exe";

    private double lambda_f, lambda_x, lambda_w;
    private double K;
    private int predLen;
    private int ar_order;
    private int stride;
    private static final double defaultLambdaF = 0.5, defaultLambdaX = 0.5, defaultLambdaW = 128;
    private static final double defaultK = 3;
    private static final int default_ar_order = 120;
    private static final int default_stride = 1;
    private ArrayList<Integer> lag_set;

    public trmf(){
        super(new HashMap<>());
    }
    public trmf(HashMap<String, ArrayList<String>> in) {
        super(in);
    }



    private ArrayList<Integer> lag_set(Integer ar_order) {
        if(ar_order == null) {
            //return new ArrayList<>(Arrays.asList(new Integer[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191}));
            return new ArrayList<>(Arrays.asList(new Integer[]{1,2,3,4,5,6,7,8,9,10}));
        }
        ArrayList<Integer> ret = new ArrayList<>();
        for(int i = 1; i<=ar_order;i++){
            ret.add(i);
        }
        return ret;
    }

    public trmf(double lambda_f, double lambda_w, double lambda_x, double k){
        this.lambda_f = lambda_f;
        this.lambda_w = lambda_w;
        this.lambda_x = lambda_x;
        this.K = k;

    }
    @Override
    protected Output evaluate(){
        System.out.println(stepSize);
        try {
            new File("Scripts\\C\\Output\\TRMF_runtime.txt").delete();
            new TimeSeries(ts).writeToCSV(dataPath);
            System.out.println("k = "+K);
            K = K < 1 ? (int) Math.ceil(ts.getRowDimension() * K) : K;

            System.out.println("k = "+K);

            if(K>ts.getRowDimension()){
                o.invalidate();
                return o;
            }

            update_stepSize(o.getTest().getColumnDimension());
            initWindowSizeAndnbWindows(o.getTest().getColumnDimension());

            //initWindowSizeAndnbWindows(o.getTest().getColumnDimension());
            executeTestAccuracyScript();
            BufferedReader reader = new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\TRMF_runtime.txt")));
            o.setRunTime(new Double(reader.readLine()));
            o.setForecast(new TimeSeries("Scripts\\C\\Output\\TRMF_out.csv").toMatrix().getSubMatrix(0, o.getTest().getRowDimension() - 1, 0, o.getTest().getColumnDimension() - 1));
        }  catch (IOException e) {
            e.printStackTrace();
        }
        return o;

    }

    @Override
    public String toString() {
        return "trmf";
    }

    private void initWindowSizeAndnbWindows(int columnDimension) {
        predLen = (int)Math.ceil(((double) columnDimension)/stepSize);


    }

    public HashMap<String,String[]> getSearchSpace(){
        HashMap<String,String[]> space = new HashMap<>();
        space.put("-lambdaf",new String[]{"0.125","0.5","2","8"});
        space.put("-lambdax",new String[]{"0.125","0.5","2","8"});
        space.put("-lambdaw",new String[]{"0.5","2","8","32","125","400","625","1000"});
        space.put("-k",new String[]{"0.02","0.05","0.1","0.15","0.2","0.3"});



        return space;
    }

    protected void executeTestAccuracyScript() {
        try {
            String s = testAccuracyScriptPath+" "+  //path to script
                    lambda_f+" "+
                    lambda_w+" "+
                    lambda_x+" "+
                    (int)K+" "+
                    predLen+" "+
                    stepSize+" "+
                    dataPath+" "+
                    ar_order+" "+
                    stride;
            System.out.println(s);
            // run the Unix "ps -ef" command
            // using the Runtime exec method:
            Process p = Runtime.getRuntime().exec(

                    s);



            BufferedReader stdInput = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            // read the output from the command
            System.out.println("Here is the standard output of the command:\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }

            // read any errors from the attempted command
            System.out.println("Here is the standard error of the command (if any):\n");
            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }


        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }

    }

    protected void executeAllHorizonScript() {
        System.out.println(ar_order);
        try {
            String s = allHorizonScriptPath+" "+  //path to script
                    lambda_f+" "+
                    lambda_w+" "+
                    lambda_x+" "+
                    (int)K+" "+
                    dataPath;
                    //(ar_order != null ? ar_order : "");
            System.out.println(s);
            // run the Unix "ps -ef" command
            // using the Runtime exec method:
            Process p = Runtime.getRuntime().exec(

                    s);



            BufferedReader stdInput = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            // read the output from the command
            System.out.println("Here is the standard output of the command:\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }

            // read any errors from the attempted command
            System.out.println("Here is the standard error of the command (if any):\n");
            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }


        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }

    }

    protected MultiOutput compute_all_horizons(MultiOutput out){
        try {
            new TimeSeries(out.getTrain()).writeToCSV(dataPath);
            K = K < 1 ? (int) Math.ceil(ts.getRowDimension() * K) : K;


            executeAllHorizonScript();
            BufferedReader reader = new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\TRMF_runtime.txt")));
            out.setRunTime(new Double(reader.readLine()));

            RealMatrix Y = new TimeSeries("Scripts\\C\\TRMF\\latent_series.csv").toMatrix();
            RealMatrix F = new TimeSeries("Scripts\\C\\TRMF\\recombiner.csv").toMatrix();
            RealMatrix W = new TimeSeries("Scripts\\C\\TRMF\\lag_values.csv").toMatrix();
            Integer OG_stepSize = stepSize;
            for(stepSize = 1;stepSize<=out.getTest().getColumnDimension();stepSize++){
                out.add_horizon(stepSize,predict(Y,F,W,out.getTest()));
            }
            stepSize = OG_stepSize;

        }  catch (IOException e) {
            e.printStackTrace();
        }
        return out;

    }

    private RealMatrix predict(RealMatrix Y, RealMatrix F, RealMatrix W, RealMatrix test) {
        RealMatrix working_Y = Y.copy();
        RealMatrix FTpseudo =  new SingularValueDecomposition(F.transpose()).getSolver().getInverse();
        RealMatrix forecasts = MatrixUtils.createRealMatrix(Y.getRowDimension(), test.getColumnDimension());
        for(int i = 0; i<test.getColumnDimension();i++) {
            if(i%stepSize==0&&i!=0){
                int len = Math.min(stepSize,working_Y.getColumnDimension());
                working_Y.setSubMatrix(FTpseudo.multiply(test.getSubMatrix(0,test.getRowDimension()-1,i-len,i-1)).getData(),0,working_Y.getColumnDimension()-len);
            }
            RealMatrix prev_values = select(working_Y, lag_set);
            RealVector pred = MatrixTools.sumColumn(MatrixTools.hadamard_prduct(W,prev_values));
            forecasts.setColumnVector(i,pred);
            working_Y = MatrixTools.columnMerge(working_Y,MatrixUtils.createColumnRealMatrix(pred.toArray()));
        }
        return F.transpose().multiply(forecasts);
    }

    private RealMatrix select(RealMatrix Y, ArrayList<Integer> lag_set) {
        RealMatrix ret = MatrixUtils.createRealMatrix(Y.getRowDimension(),lag_set.size());
        int count = 0;
        for(Integer lag : lag_set){
            ret.setColumn(count++,Y.getColumn(Y.getColumnDimension()-lag));

        }
        return ret;
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

    protected void parse(HashMap<String, ArrayList<String>> input) {

        lambda_f = input.containsKey("-lambdaf") ? new Double(input.get("-lambdaf").get(0)) : defaultLambdaF;
        lambda_x = input.containsKey("-lambdax") ? new Double(input.get("-lambdax").get(0)) : defaultLambdaX;
        lambda_w = input.containsKey("-lambdaw") ? new Double(input.get("-lambdaw").get(0)) : defaultLambdaW;
        stepSize = input.containsKey("-windowSize") ? new Integer(input.get("-windowSize").get(0)) : null;
        ar_order = input.containsKey("-ar_order") ? new Double(input.get("-ar_order").get(0)).intValue() : default_ar_order;
        stride = input.containsKey("-stride") ? new Double(input.get("-stride").get(0)).intValue() : default_stride;
        //lag_set = lag_set(ar_order);
        K = input.containsKey("-k") ? new Double(input.get("-k").get(0)) : defaultK;



    }


}
