package thesis.Tools;

import org.apache.commons.math3.linear.MatrixUtils;
import thesis.TSModel;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

public class MultiOutputParser {
    public static void parseMultiOutput(TSModel.MultiOutput o, BufferedReader reader) throws IOException {
        o.setRunTime(Double.parseDouble(reader.readLine()));
        String all = reader.lines().collect(Collectors.joining(System.lineSeparator()));
        String[] horizon_forecast = all.split("#");
        for(String h_f : horizon_forecast){
            //System.out.println(h_f);
            ArrayList<Double[]> matrix_temp = new ArrayList<>();
            Double[] row_temp;
            String[] lines = h_f.split(System.lineSeparator());
            lines = drop_empty(lines);
            int horizon = new Integer(lines[0]);
            for(int i = 1; i<lines.length;i++){
                String[] row = lines[i].split(" ");
                row_temp = new Double[row.length];
                for(int j = 0; j<row.length;j++){
                    row_temp[j] =  row[j].equalsIgnoreCase("nan")||row[j].equalsIgnoreCase("-inf")||row[j].equalsIgnoreCase("inf") ? Double.NaN : new Double(row[j]);
                }
                matrix_temp.add(row_temp);
            }
            o.add_horizon(horizon,MatrixUtils.createRealMatrix(ArrayUtils.toPrimitive(matrix_temp.stream().toArray(Double[][]::new))));
        }
    }

    private static String[] drop_empty(String[] lines) {
        ArrayList<String> temp = new ArrayList<>();
        String[] temptemp = new String[0];
        for(String s : lines){
            if(!s.equalsIgnoreCase(""))
                temp.add(s);
        }
        return temp.toArray(temptemp);
    }
}
