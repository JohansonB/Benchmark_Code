package thesis.Models;

import thesis.TSModel;
import thesis.Tools.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public abstract class DartsModel extends TSModel {
    private final String interpreterpath = "C:\\Users\\41766\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.exe";
    private final String scriptPathTestAccuracy = "Scripts\\Python\\Darts\\DartsModel.py";
    private final String scriptPathAllHorizons = "Scripts\\Python\\Darts\\DartsAllHorizon.py";
    private String pathToTs = "Scripts\\Python\\temp\\"+toString()+"_trainIn.csv";
    private String pathToTest = "Scripts\\Python\\temp\\"+toString()+"_testIn.csv";
    String additionalArguments;
    private double output_length;
    protected double input_chunk;
    protected boolean normalize;

    protected static double default_output_length = 0.1;
    protected static double default_input_length = 0.3;

    private static class Error_Reader extends Thread{
        protected String error_mesage;
        BufferedReader error_stream;

        Error_Reader(BufferedReader error_stream){
            this.error_stream = error_stream;
        }

        public void run(){
            String s;
            StringBuilder sb = new StringBuilder();
            try {
                while ((s = error_stream.readLine()) != null) {
                    sb.append(s+"\n");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            error_mesage = sb.toString();

        }
    }

    public DartsModel(HashMap<String, ArrayList<String>> in) {
        super(in);
        update(in);
    }
    protected void update(HashMap<String, ArrayList<String>> in){
        normalize = in.containsKey("-normalize") ? new Boolean(in.get("-normalize").get(0)) : true;


        output_length = in.containsKey("-output_len") ? new Double(in.get("-output_len").get(0)) : default_output_length;
        input_chunk = in.containsKey("-input_chunk") ? new Double(in.get("-input_chunk").get(0)) : default_input_length;
    }
    public DartsModel() {
        super();
    }

    //gives the childs the chance to update its params based upon knoledge about the test and train set.
    protected void update_params_context(){

    }
    @Override
    protected Output evaluate(){
        update_params_context();
        try {
            new File("Scripts\\Python\\Output\\"+ toString() + "_Out.txt").delete();
            System.out.println("Test dimenison: "+MatrixTools.getDimensions(o.getTest()));
            new TimeSeries(o.getTrain()).writeToCSV(pathToTs);
            new TimeSeries(o.getTest()).writeToCSV(pathToTest);
            executeScriptTestAccuracy();
            OutputParser.parseOutput(o, new BufferedReader(new FileReader(new File("Scripts\\Python\\Output\\"+ toString() + "_Out.txt"))));
        }  catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return o;
    }

    protected MultiOutput compute_all_horizons(MultiOutput out){
        if(!is_nn()){
            throw new Error("Operation is only supported for neural network models");
        }
        try {
            new TimeSeries(out.getTrain()).transpose().writeToCSV(pathToTs);
            new TimeSeries(out.getTest()).transpose().writeToCSV(pathToTest);
            executeScriptAllHorizons(out);
            MultiOutputParser.parseMultiOutput(out, new BufferedReader(new FileReader(new File("Scripts\\Python\\Output\\"+ toString() + "_Out.txt"))));
        }  catch (IOException e) {
            e.printStackTrace();
        }
        return out;
    }

    protected abstract boolean is_nn();

    /*public Output test_accuracy(RealMatrix ts, double split){
          /*if(tune_hyperParams) {
            GridSearcher.optimize(this,o.getTrain(), toString());
            tune_hyperParams = false;
        }
        this.ts = ts;
        this.split = split;
        o = new Output(ts, split);
        update_stepSize(o.getTest().getColumnDimension());
        return evaluate();


    }*/
    private void executeScriptTestAccuracy(){
        try {
            String s= interpreterpath+" "+scriptPathTestAccuracy+" --train_path "+pathToTs+" --test_path "+pathToTest+" --algo "+toString()+" --output_len "+(output_length <1 ?(int)(o.getTrain().getColumnDimension()*output_length) : (int)output_length)+" --input_chunk "+(input_chunk < 1 ?(int)(o.getTrain().getColumnDimension()*input_chunk) : (int)input_chunk)+" --normalize "+(normalize ? "True" : "False")+" "+(additionalArguments==null ? "" : additionalArguments);
            System.out.println(s);
            Process p = Runtime.getRuntime().exec(s);
            StringBuilder ErrorOutput;

            BufferedReader stdInput = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            //stdInput.close();
            //stdError.close();
            //p.waitFor();
            // read the output from the command
            Error_Reader err = new Error_Reader(stdError);
            err.start();
            System.out.println("Here is the output (if any):\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }
            err.join();
            System.out.println("Here is the Error (if any):\n");
            System.out.println(err.error_mesage);











        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        } catch (InterruptedException e) {
            System.out.println("Got interrupted although this should never happen");
            e.printStackTrace();
            System.exit(-1);
        }
    }
    private void executeScriptAllHorizons(MultiOutput out){
        try {
            String s= interpreterpath+" "+scriptPathAllHorizons+" --train_path "+pathToTs+" --test_path "+pathToTest+" --algo "+toString()+" --output_len "+(output_length < 1 ?(int)(out.getTest().getColumnDimension()*output_length) : (int)output_length)+" "+(additionalArguments==null ? "" : additionalArguments);
            System.out.println(s);
            Process p = Runtime.getRuntime().exec(s);


            //p.waitFor();

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
    public void updatePath(){
        pathToTs = "Scripts\\Python\\temp\\"+toString()+"_in.csv";
    }   



    @Override
    public boolean is1D() {
        return true;
    }

    @Override
    public boolean isND() {
        return is_nn();
    }

    @Override
    public boolean missingValuesAllowed() {
        return false;
    }

    public HashMap<String,String[]> getSearchSpace(){
        return new HashMap<>();
    }


}
