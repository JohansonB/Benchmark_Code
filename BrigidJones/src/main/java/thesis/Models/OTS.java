package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import thesis.TSModel;
import thesis.Tools.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class OTS extends TSModel {
    private final String testAccuracyScriptPath = "Scripts\\C\\OTS\\OTSmain.exe";
    private final String allHorizonScriptPath = "Scripts\\C\\OTS\\OTS_Model.exe";
    private String dataPath = "Scripts\\C\\Temp\\OTS_in.csv";

    public OTS(){
        super(new HashMap<>());
    }

    public OTS(HashMap<String, ArrayList<String>> in) {
        super(in);
    }

    protected Output evaluate(){

        MatrixTools.setRange(ts,o.getTrain().getColumnDimension(),ts.getColumnDimension()-1,Double.NaN);
        try {
            new File("Scripts\\C\\Output\\OTS_runtime.txt").delete();
            new TimeSeries(ts).writeToCSV(dataPath);
            executeTestAccuracyScript();
            BufferedReader reader = new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\OTS_runtime.txt")));
            o.setRunTime(new Double(reader.readLine()));
            RealMatrix temp;
            o.setForecast((temp=new TimeSeries("Scripts\\C\\Output\\OTS_out.csv").toMatrix()).getSubMatrix(0,temp.getRowDimension()-1,temp.getColumnDimension()-o.getTest().getColumnDimension(),temp.getColumnDimension()-1));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return o;
    }

    protected MultiOutput compute_all_horizons(MultiOutput out){
        try {
            new TimeSeries(out.getTrain()).writeToCSV(dataPath);
            executeAllHorizonScript();
            BufferedReader reader = new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\OTS_runtime.txt")));
            out.setRunTime(new Double(reader.readLine()));
            RealMatrix weights = new TimeSeries("Scripts\\C\\Output\\OTS_weights.csv").toMatrix();
            //weights = MatrixTools.reverse(weights);
            RealMatrix train = new TimeSeries("Scripts\\C\\Output\\OTS_out.csv").toMatrix();
            Integer OG_stepSize = stepSize;
            for(stepSize=1;stepSize<=out.getTest().getColumnDimension();stepSize++){
                out.add_horizon(stepSize,predict(weights,train,out.getTest()));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return out;

    }

    private RealMatrix predict(RealMatrix weights,RealMatrix train, RealMatrix test) {
        weights = MatrixTools.reverse(weights);
        RealMatrix past_values = train.getSubMatrix(0,train.getRowDimension()-1,train.getColumnDimension()-weights.getColumnDimension()+1,train.getColumnDimension()-1);
        RealMatrix lag_matrix = MatrixUtils.createRealMatrix(past_values.getRowDimension(),past_values.getColumnDimension()+1);
        lag_matrix.setColumn(lag_matrix.getColumnDimension()-1,ArrayUtils.create(past_values.getRowDimension(),1));
        lag_matrix.setSubMatrix(past_values.getData(),0,0);
        RealMatrix ret = MatrixUtils.createRealMatrix(test.getRowDimension(),test.getColumnDimension());
        for(int i = 0; i< test.getColumnDimension();i++){
            if(i%stepSize==0&&i!=0){
                int len = Math.min(stepSize,past_values.getColumnDimension());
                past_values.setSubMatrix(test.getSubMatrix(0,test.getRowDimension()-1,i-len,i-1).getData(),0,past_values.getColumnDimension()-len);
                lag_matrix.setSubMatrix(past_values.getData(),0,0);
            }
            RealVector pred = MatrixTools.sumColumn(MatrixTools.hadamard_prduct(lag_matrix,weights));
            ret.setColumnVector(i,pred);
            past_values.setSubMatrix(past_values.getSubMatrix(0,past_values.getRowDimension()-1,1,past_values.getColumnDimension()-1).getData(),0,0);
            past_values.setColumnVector(past_values.getColumnDimension()-1,pred);
            lag_matrix.setSubMatrix(past_values.getData(),0,0);
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
        return true;
    }

    @Override
    public String toString(){
        return "ots";
    }

    public enum Opt{
        OGD,
        GSR;
    }


    Opt op = Opt.OGD;


    private  final static double defaultar_order = 0.5;
    private  final static double defaultGamma = 0.00000001;

    private double order;
    private double gamma;





    public HashMap<String,String[]> getSearchSpace(){
        System.out.println("hoi");
        HashMap<String,String[]> space = new HashMap<>();
        space.put("-order",new String[]{"5","10","20","25"});
        space.put("-gamma",new String[]{"0.00001","0.0001","0.1","1","10"});


        return space;
    }
    protected void executeTestAccuracyScript() {
        try {
            if(order<1){
                //ratio convert to absolut
                order = o.getTrain().getColumnDimension()*order;
            }
            String s = testAccuracyScriptPath+" "+  //path to script
                    (int)order+" "+
                    gamma+" "+
                    (op == Opt.OGD ? 0 : 1)+" "+
                    dataPath;
            System.out.println(s);
            // run the Unix "ps -ef" command
            // using the Runtime exec method:
            Process p = Runtime.getRuntime().exec(s);



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
        try {

            String s = allHorizonScriptPath+" "+  //path to script
                    order+" "+
                    gamma+" "+
                    dataPath;
            System.out.println(s);
            // run the Unix "ps -ef" command
            // using the Runtime exec method:
            Process p = Runtime.getRuntime().exec(s);



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
    public String getParams(){
        return "alpha: "+gamma+"\norder: "+order;
    }
    protected void parse(HashMap<String, ArrayList<String>> input) {
        order = input.containsKey("-order") ? new Double(input.get("-order").get(0)) : defaultar_order;
        gamma = input.containsKey("-gamma") ? new Double(input.get("-gamma").get(0)) : defaultGamma;
        if(input.containsKey("-version")){
            String temp = input.get("-version").get(0);
            if(temp.equalsIgnoreCase("ogd"))
                op = Opt.OGD;
            else if(temp.equalsIgnoreCase("gsr"))
                op = Opt.GSR;
            else{
                throw new IllegalArgumentException(temp +" is not an optimizer only [ogd, gsr] available");
            }
        }


    }



}
