package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import thesis.TSModel;
import thesis.Tools.ArrayUtils;

import java.util.ArrayList;
import java.util.HashMap;

public class Naive extends TSModel {
    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {

    }

    @Override
    protected Output evaluate() {
        int j;
        double y_hat = 0;
        o.setForecast(MatrixUtils.createRealMatrix(o.getTest().getRowDimension(),o.getTest().getColumnDimension()));
        for(int i = 0; i<o.getTrain().getRowDimension();i++){
            j = o.getTrain().getRowDimension()-1;
            while(j>=0&&Double.isNaN(o.getTrain().getEntry(i,j)))
                j--;
            if(i!=0 || !Double.isNaN(o.getTrain().getEntry(i,j)))
                y_hat = o.getTrain().getEntry(i,j);
            o.getForecast().setColumn(i,ArrayUtils.create(o.getForecast().getColumnDimension(),y_hat));
            y_hat = 0;

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
        return true;
    }

    @Override
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }
}
