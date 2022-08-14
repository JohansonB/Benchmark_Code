package thesis.DataGenerators;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import thesis.Tools.TimeSeries;

import java.io.IOException;

public class Solar_to_hourly {
    public static void convert_and_store() throws IOException {
        RealMatrix data_mat = new TimeSeries("Datasets\\solar_10_minutes_dataset.csv").toMatrix();
        RealMatrix result = MatrixUtils.createRealMatrix(data_mat.getRowDimension(),data_mat.getColumnDimension()/36);
        for(int i = 0; i<data_mat.getRowDimension();i++){
            int count = 0;
            int count_count = 0;
            double tot = 0;
            for(int j = 0; j<data_mat.getColumnDimension();j++){
                tot += data_mat.getEntry(i,j);
                count += 1;
                if(count%36 == 0){
                    result.setEntry(i,count_count++,tot);
                    count = 0;
                    tot = 0;
                }


            }
        }
        new TimeSeries(result).writeToCSV("Datasets\\solar_6_hourly_dataset.csv");

    }
    public static void energy_convert_and_store() throws IOException {
        RealMatrix data_mat = new TimeSeries("Datasets\\electricity_hourly_dataset.csv").toMatrix();
        RealMatrix result = MatrixUtils.createRealMatrix(data_mat.getRowDimension(),(data_mat.getColumnDimension()-1)/6);
        for(int i = 0; i<data_mat.getRowDimension();i++){
            int count = 0;
            int count_count = 0;
            double tot = 0;
            for(int j = 1; j<data_mat.getColumnDimension();j++){
                tot += data_mat.getEntry(i,j);
                count += 1;
                if(count%6 == 0){
                    result.setEntry(i,count_count++,tot);
                    count = 0;
                    tot = 0;
                }


            }
        }
        new TimeSeries(result).writeToCSV("Datasets\\electricity_6hourly_dataset.csv");

    }
    public static void main(String[] args) throws IOException {
        energy_convert_and_store();
    }
}
