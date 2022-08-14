package thesis.Tools;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.io.IOException;

public class AutoCorrelation {
    public RealVector auto_corelation_function;
    public AutoCorrelation(RealMatrix r, int max_lag){
        auto_corelation_function = compute_auto_core(r,max_lag);

    }

    private RealVector compute_auto_core(RealMatrix r,int max_lag) {
        RealMatrix temp;
        RealMatrix correlation;
        RealVector auto_func = MatrixUtils.createRealVector(new double[max_lag]);
        for(int i = 1; i<=max_lag;i++){
            temp = MatrixUtils.createRealMatrix(2,r.getColumnDimension()-i);
            temp.setRowVector(0,r.getRowVector(0).getSubVector(0,r.getColumnDimension()-i));
            temp.setRowVector(1,MatrixTools.leftShift(r,i).getRowVector(0));
            correlation = new PearsonsCorrelation().computeCorrelationMatrix(temp.transpose());
            auto_func.setEntry(i-1,correlation.getEntry(0,1));
        }
        return auto_func;
    }
    private TimeSeries fft(){
        FastFourierTransformer f = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] res = f.transform(auto_corelation_function.toArray(), TransformType.INVERSE);
        double[] ret = new double[res.length];
        for(int i = 0; i<res.length;i++){
            ret[i] = res[i].abs();
        }
        return new TimeSeries(ret);
    }

}
