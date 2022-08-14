package thesis.Models;


import thesis.TSModel;
import thesis.Tools.GridSearcher;
import thesis.Tools.OutputParser;
import thesis.Tools.TimeSeries;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class Prophet extends TSModel {
    private static final String trainPath = "Scripts\\R\\Prophet\\train.csv";
    private static final String testPath = "Scripts\\R\\Prophet\\test.csv";
    private static final String outputPath = "Scripts\\R\\Prophet\\out.csv";
    private static final String interpreterpath = "C:\\Program Files\\R\\R-4.1.0\\bin\\Rscript.exe";
    private static final String scriptPath = "Scripts\\R\\Prophet\\prophet2.R";


    private double costum_period;
    private String start_date;
    private String date_step;
    private String trend;

    public Prophet(HashMap<String, ArrayList<String>> input) {
        super(input);
    }

    public Prophet() {

    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        costum_period = in.containsKey("-period") ? new Double(in.get("-period").get(0)) : 0;
        start_date = in.containsKey("-start_date") ? in.get("-start_date").get(0) : "1/1/2000";
        date_step = in.containsKey("-date_step") ? in.get("-date_step").get(0).replaceAll(" ","_") : "day";
        trend = in.containsKey("-trend") ? in.get("-trend").get(0) : "flat";

    }

    @Override
    protected Output evaluate() {

        try {
            OutputParser.clear("R","Prophet");
            new TimeSeries(o.getTrain()).writeToCSV(trainPath);
            executeScript();
            o = new OutputParser("R","Prophet").readOutput(o);
        }  catch (IOException e) {
            e.printStackTrace();
        }
        return o;
    }
    private void executeScript(){
        try {
            String s= interpreterpath+" "+scriptPath+" "+o.getTest().getColumnDimension()+" "+start_date+" "+date_step+" "+trend;
            System.out.println(s);
            Process p = Runtime.getRuntime().exec(s);


            p.waitFor();

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
        catch (IOException | InterruptedException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }

    }

    @Override
    public boolean is1D() {
        return true;
    }

    public String toString(){
        return "prophet";
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
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }


}
