package thesis.Models;

import thesis.TSModel;
import thesis.Tools.*;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ME extends TSModel {
    protected String testAcurracyScriptPath = "Scripts\\Python\\ME\\ME.py";
    protected String allHorizonScriptPath = "Scripts\\Python\\ME\\MEAllHorizon.py";
    private String pathToTrain = "Scripts\\Python\\ME\\train.csv";
    private String pathToTest = "Scripts\\Python\\ME\\test.csv";
    protected static final String interpreterpath = "C:\\Users\\41766\\AppData\\Local\\Programs\\Python\\Python38\\python.exe";
    private static double defaultCompressionRate = 0.95;
    private static double defaultMNRatio = 1;
    private static int defaultN = 81;

    private double compressionRate;
    private double MNRatio;
    private int n;


    public ME(HashMap<String, ArrayList<String>> input) {
        super(input);
    }
    public ME() {

    }





    protected void executeTestAcurracyScript() {
        try {
            new TimeSeries(o.getTrain()).writeToCSV(pathToTrain);
            new TimeSeries(o.getTest()).writeToCSV(pathToTest);
            String s = interpreterpath+" " + //path to python exe
                    testAcurracyScriptPath+" "+  //path to script
                    "--train_root "+pathToTrain+" "+
                    "--test_root "+pathToTest+" "+
                    "--windowSize "+stepSize+" "+
                    "--compression_ratio "+ compressionRate+" "+
                    " --MN_ratio "+ MNRatio+
                    " --n "+ n;
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
            new TimeSeries(o.getTrain()).writeToCSV(pathToTrain);
            new TimeSeries(o.getTest()).writeToCSV(pathToTest);
            String s = interpreterpath+" " + //path to python exe
                    allHorizonScriptPath+" "+  //path to script
                    "--train_root "+pathToTrain+" "+
                    "--test_root "+pathToTest+" "+
                    "--compression_ratio "+ compressionRate+" "+
                    " --MN_ratio "+ MNRatio+
                    " --n "+ n;
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


    //Overrides preconfigured parameters n and m and sets them automatically
    @Override
    protected TSModel.Output evaluate(){
        new File("Scripts\\Python\\ME\\out.txt").delete();
        executeTestAcurracyScript();
        try {
            OutputParser.parseOutput(o,new BufferedReader(new FileReader(new File("Scripts\\Python\\ME\\out.txt"))));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        return o;
    }

    protected MultiOutput compute_all_horizons(MultiOutput out){
        executeAllHorizonScript();
        try {
            MultiOutputParser.parseMultiOutput(out,new BufferedReader(new FileReader(new File("Scripts\\Python\\ME\\out.txt"))));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return out;

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
        return "MESVD";
    }

    @Override
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }

    protected void parse(HashMap<String, ArrayList<String>> input){
        compressionRate = input.containsKey("-compressionRate") ? new Double(input.get("-compressionRate").get(0)) : defaultCompressionRate;
        MNRatio = input.containsKey("-MNRatio") ? new Double(input.get("-MNRatio").get(0)) : defaultMNRatio;
        n = input.containsKey("-n") ? new Double(input.get("-n").get(0)).intValue() : defaultN;
    }


    private static void optimus_prime(HashMap<String,String> init,int low, int high,String data_path) throws IOException {
        HashMap<String,String> h = (HashMap<String, String>) init.clone();
        ArrayList<Double> error_temp;
        ArrayList<Double> comperssion_temp;
        ArrayList<Double> final_errors = new ArrayList<>();
        ArrayList<Double> final_compression = new ArrayList<>();
        ME mod = new ME(GridSearcher.reformat(h));
        for(int i = low;i<=high;i++){
            System.out.println(i);
            error_temp = new ArrayList<>();
            comperssion_temp = new ArrayList<>();
            h.put("-n",new Integer(i).toString());
            Vary_Result v_r = mod.vary_parameter(data_path,1,40,1,"-compressionRate",h);
            for (Map.Entry<Double, Output> entry : v_r.get_results().entrySet()) {
                Double k = entry.getKey();
                Output v = entry.getValue();
                error_temp.add(v.error(new MAE()));
                comperssion_temp.add(k);
            }
            int index = error_temp.indexOf(Collections.min(error_temp));
            final_errors.add(error_temp.get(index));
            final_compression.add(comperssion_temp.get(index));


        }
        //storing wanted output just in case i mess up the plotting
        ArrayList<String> errors = new ArrayList<>();
        ArrayList<String> compression = new ArrayList<>();
        final_errors.forEach(v -> errors.add(v.toString()));
        final_compression.forEach(v -> compression.add(v.toString()));
        Path error_out = Paths.get("C:\\Users\\41766\\Desktop\\Baechlor_Sourceroni_code\\temp\\error.txt");
        Path compression_out = Paths.get("C:\\Users\\41766\\Desktop\\Baechlor_Sourceroni_code\\temp\\compression.txt");
        Files.write(error_out,errors, Charset.defaultCharset());
        Files.write(compression_out,compression,Charset.defaultCharset());

        int count = 0;
        Graph g1 = new Graph(data_path, "vary_n_error", "-n", "MAE",true);
        Graph g2 = new Graph(data_path, "vary_n_compression", "-n", "compression_dim",true);
        for(int i = low;i<=high;i++){
            g1.add("a",i,final_errors.get(count));
            g2.add("a",i,final_compression.get(count++));
        }
        g1.plot();
        g2.plot();
    }

}
