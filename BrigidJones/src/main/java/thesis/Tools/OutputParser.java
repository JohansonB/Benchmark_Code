package thesis.Tools;

import org.apache.commons.math3.linear.MatrixUtils;
import thesis.TSModel;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class OutputParser {
    private String language,  algorithm;
    private BufferedReader seriesReader;
    private BufferedReader runtimeReader;
    public OutputParser(String language, String algorithm){
        this.language = language;
        this.algorithm = algorithm;
        try {
            runtimeReader = new BufferedReader(new FileReader(new File("Scripts\\"+language+"\\"+algorithm+"\\runtime.txt")));
            seriesReader = new BufferedReader(new FileReader(new File("Scripts\\"+language+"\\"+algorithm+"\\out.csv")));

        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public static void clear(String language, String algorithm) {
        new File("Scripts\\"+language+"\\"+algorithm+"\\runtime.txt").delete();
        new File("Scripts\\"+language+"\\"+algorithm+"\\out.csv").delete();

    }

    public TSModel.Output readOutput(TSModel.Output o){
        try {
            o.setRunTime(new Double(runtimeReader.readLine()));
            ArrayList<String> string_list = new ArrayList<>();
            ArrayList<Double> double_list = new ArrayList<>();
            seriesReader.lines().forEach(s -> string_list.add(s));
            for(String s : string_list){
                if(isNumeric(s))
                    double_list.add(new Double(s));
            }
            Double[] temp = new Double[0];
            temp = double_list.toArray(temp);
            System.out.println(temp);
            o.setForecast(new TimeSeries(temp).toMatrix());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return o;
    }

    private void drop_non_numeric(ArrayList<String> elts) {
        for(String s : elts){
            if(!isNumeric(s))
                elts.remove(s);
        }
    }
    private static boolean isNumeric(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            double d = Double.parseDouble(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    public static void parseOutput(TSModel.Output o, BufferedReader reader) throws IOException {
        o.setRunTime(Double.parseDouble(reader.readLine()));
        ArrayList<Double[]> temp = new ArrayList<>();
        reader.lines().forEach(x -> temp.add(Arrays.stream(x.split(" ")).map(y->y.equalsIgnoreCase("nan")||y.equalsIgnoreCase("-inf")||y.equalsIgnoreCase("inf") ? Double.NaN : new Double(y)).toArray(Double[]::new)));

        o.setForecast(MatrixUtils.createRealMatrix(ArrayUtils.toPrimitive(temp.stream().toArray(Double[][]::new))));
    }

    private static double[] parseArray(String x) {
        AtomicInteger charCount = new AtomicInteger();
        AtomicInteger tempCount = new AtomicInteger();
        ArrayList<Double> temp = new ArrayList<>();
        char[]  number = new char[x.length()];


        ArrayList<Double> finalTemp = temp;
        x.chars().forEach(c -> {
            if (' ' == c) finalTemp.add(Double.parseDouble(new String(number, 0, charCount.getAndSet(0))));
            else number[charCount.getAndIncrement()] = (char) c;
        });

        return temp.stream().mapToDouble(i -> i).toArray();


    }


}

