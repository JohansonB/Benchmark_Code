package thesis.Tools;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import thesis.Plotting.Plotter;


import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class TimeSeries {
    private double[][] values;
    private int dimension;

    //keep track of rescale information
    double[] mins;
    double[] maxs;


    public TimeSeries(Double[] array) {
        dimension=1;
        values = ArrayUtils.to2D(ArrayUtils.toPrimitive(array));

    }
    public TimeSeries(Double[][] array) {
        values = ArrayUtils.toPrimitive(array);
        checkDimension(values);

    }

    public TimeSeries(RealVector pred) {
        dimension = 1;
        values = ArrayUtils.to2D(pred.toArray());
    }

    public TimeSeries(ArrayList<Double> ts) {
        Double[] data = new Double[0];
        data = ts.toArray(data);
        values = ArrayUtils.to2D(ArrayUtils.toPrimitive(data));
    }


    public Double[][] toArray(){
        return ArrayUtils.toObject(values);
    }
    public TimeSeries(double[][] values){
        checkDimension(values);
        this.values = values;
    }

    public TimeSeries(RealMatrix r){
        values = ArrayUtils.transpose(r.getData());
        dimension = r.getRowDimension();
    }
    public TimeSeries(){
        dimension = 0;
        values = new double[0][];
    }
    public int getDimension(){
        return dimension;
    }
    private void checkDimension(double[][] v) throws IllegalTimeSeries {
        if(v==null){
            return;
        }
        dimension = v[0].length;
        for (int i = 1; i<v.length;i++){
            if(v[i].length!=dimension){
                throw new IllegalTimeSeries("all elements of the Time series need to have the same dimension");
            }
        }
    }
    public TimeSeries(double[] data){
        dimension=1;
        values = ArrayUtils.to2D(data);
    }
    public TimeSeries(String csvPath) throws IOException {
        String row;
        BufferedReader csvReader = new BufferedReader(new FileReader(csvPath));
        ArrayList<double[]> temp = new ArrayList<>();
        while ((row = csvReader.readLine()) != null) {
            String[] data = row.split(",");
            double[] val = new double[data.length];
            for(int i = 0; i<data.length;i++){
                val[i] = data[i].equalsIgnoreCase("nan")||data[i].equalsIgnoreCase("inf")||data[i].equalsIgnoreCase("-inf")||data[i].equalsIgnoreCase("") ? Double.NaN : Double.parseDouble(data[i]);
            }
            temp.add(val);
        }
        csvReader.close();
        values = new double[temp.size()][];
        for(int i = 0; i<temp.size();i++){
            values[i] = temp.get(i);
        }
        values = ArrayUtils.transpose(values);
        checkDimension(values);

    }
    public TimeSeries append(double[] ... in){
        double[][] temp = new double[values.length+in.length][];
        int i = values.length;
        for(double[] d : in){
            temp[i++] = d;
        }
        System.arraycopy(values,0,temp,0,values.length);
        values = temp;
        return this;
    }
    public TimeSeries append(Double[] ... in){
        double[][] temp = new double[values.length+in.length][];
        int i = values.length;
        for(Double[] d : in){
            temp[i++] = ArrayUtils.toPrimitive(d);
        }
        System.arraycopy(values,0,temp,0,values.length);
        values = temp;
        return this;
    }
    public void append(double[][] ... in){
        int tot = 0;
        for(double[][] i : in){
            tot+=i.length;
        }
        double[][] temp = new double[values.length+tot][];
        int cur = values.length;
        System.arraycopy(values,0,temp,0,values.length);
        for(double[][] i : in){
            System.arraycopy(i,0,temp,cur,i.length);
            cur += i.length;
        }
        values= temp;
    }
    public void append(Double[][] ... in){
        int tot = 0;
        for(Double[][] i : in){
            tot+=i.length;
        }
        double[][] temp = new double[values.length+tot][];
        int cur = values.length;
        System.arraycopy(values,0,temp,0,values.length);
        for(Double[][] i : in){
            System.arraycopy(ArrayUtils.toPrimitive(i),0,temp,cur,i.length);
            cur += i.length;
        }
        values= temp;
    }
    public void update(int index, double[] value){
        if(dimension!=value.length){
            throw new IllegalArgumentException("dimension of input array must be :"+ dimension);
        }
        values[index] = value;
    }
    public void update(int index, Double[] value){
        if(dimension!=value.length){
            throw new IllegalArgumentException("dimension of input array must be :"+ dimension);
        }
        values[index] = ArrayUtils.toPrimitive(value);
    }
    public void append(ArrayList<Double[][]> in){
        Double[][][] temp = new Double[in.size()][][];
        int count = 0;
        for(Double[][] i : in){
            temp[count++] = i;
        }
        append(temp);
    }

    public Double[] get(int index){
        return ArrayUtils.toObject(values[index]);
    }
    public TimeSeries getSubSeries(int startIndex, int length){
        if(length==0){
            return new TimeSeries();
        }
        double[][] returner = new double[length][dimension];
        System.arraycopy(values,startIndex,returner,0,length);
        return new TimeSeries(returner);
    }
    public TimeSeries getSubSeriesDeepCopy(int startIndex, int length){
        if(length==0){
            return new TimeSeries();
        }
        double[][] returner = new double[length][];
        for(int i = 0; i<length;i++){
            double[] arr = new double[dimension];
            System.arraycopy(values[startIndex+i],0,arr,0,dimension);
            returner[i] = arr;
        }
        return new TimeSeries(returner);
    }
    public double[] getRow(int index){
        double[] ret = new double[values.length];
        for(int i = 0; i<values.length;i++){
            ret[i] = values[i][index];
        }
        return ret;
    }
    public TimeSeries get1DSubSeries(int index){
        return new TimeSeries(getRow(index));
    }
    public int length(){
        return values.length;
    }
    public Double[] mean(){
        Double[] tot = new Double[dimension];
        for(int i = 0; i<tot.length;i++){
            tot[i] = 0.;
        }
        for(int i = 0; i<values.length;i++){
            for(int j = 0; j<dimension;j++){
                tot[j] += values[i][j];
            }
        }
        for(int i = 0; i<dimension;i++){
            tot[i] = tot[i]/length();
        }
        return tot;
    }
    public Double[] mean_dev(){
        Double[] mean = mean();
        Double[] tot = new Double[dimension];
        for(int i = 0; i<dimension;i++){
            tot[i] = 0.;
        }
        for(int i= 0; i<length();i++){
            for(int j = 0;j<dimension;j++){
                tot[j] += Math.pow(values[i][j]-mean[j],2);
            }
        }
        for(int i = 0; i<dimension;i++){
            tot[i] = tot[i]/length();
        }
        return tot;
    }
    public double mean_flatten(){
        Double[] flat = flatten();
        double tot = 0;
        for(int i = 0; i<flat.length;i++){
            tot+=flat[i];
        }
        return tot/flat.length;
    }
    public double mean_dev_flatten(){
        Double[] flat = flatten();
        double tot = 0;
        double mean = mean_flatten();
        for(int i = 0; i<flat.length;i++){
            tot+=Math.pow(flat[i]-mean,2);
        }
        return tot/flat.length;
    }
    public Double[] flatten(){
        Double[] returner = new Double[dimension*length()];
        for(int i = 0; i<length();i++){
            for(int j = 0; j<dimension;j++){
                returner[dimension*i+j] = values[i][j];
            }
        }
        return returner;
    }
    public boolean isEmpty(){
        if(values==null){
            return true;
        }
        return values.length==0;
    }
    public TimeSeries reverse(){
        double[][] ret = new double[values.length][];
        for(int i = 0; i<ret.length;i++){
            ret[i] = values[values.length-1-i].clone();
        }
        return new TimeSeries(ret);
    }
    public TimeSeries difference(int order){
        if(order<0){
            throw new IllegalArgumentException("the input argument is not allowed to be negative");
        }
        return difference(order,getSubSeriesDeepCopy(0,length()));
    }
    private TimeSeries difference(int order, TimeSeries series ){
        if(order == 0){
            return series;
        }
        Double[] mean = series.mean();
        for(int i = series.length()-1;i>0;i--){
            for(int j = 0; j<series.dimension;j++){
                series.values[i][j] = series.values[i][j] - series.values[i-1][j];
            }
        }
        for(int j = 0; j<series.dimension;j++){
            series.values[0][j] = series.values[0][j]-mean[j];
        }
        return difference(order-1,series);
    }
    public TimeSeries transpose(){
        return new TimeSeries(ArrayUtils.transpose(values));
    }
   public Tuple<TimeSeries,double[][]> differenceAndLastOrders(int order){
       if(order<0){
           throw new IllegalArgumentException("the input argument is not allowed to be negative");
       }
       return differenceAndLastOrders(order,getSubSeriesDeepCopy(0,length()),new double[order][]);
   }
   private Tuple<TimeSeries,double[][]> differenceAndLastOrders(int order,TimeSeries series, double[][] acc){
       if(order == 0){
           return new Tuple<>(series,acc);
       }
       acc[order-1] = series.values[series.length()-1].clone();
       for(int i = series.length()-1;i>0;i--){
           for(int j = 0; j<series.dimension;j++){
               series.values[i][j] = series.values[i][j] - series.values[i-1][j];
           }
       }
       double[][] temp = new double[series.values.length-1][];
       System.arraycopy(series.values,1,temp,0,series.values.length-1);
       series.values = temp;
       return differenceAndLastOrders(order-1,series,acc);
   }
   public TimeSeries differenceDropFirst(int order){
       if(order<0){
           throw new IllegalArgumentException("the input argument is not allowed to be negative");
       }
       return differenceDrop(order,getSubSeriesDeepCopy(0,length()));
   }
    private TimeSeries differenceDrop(int order, TimeSeries series ){
        if(order == 0){
            return series;
        }
        for(int i = series.length()-1;i>0;i--){
            for(int j = 0; j<series.dimension;j++){
                series.values[i][j] = series.values[i][j] - series.values[i-1][j];
            }
        }
        double[][] temp = new double[series.values.length-1][];
        System.arraycopy(series.values,1,temp,0,series.values.length-1);
        series.values = temp;
        return differenceDrop(order-1,series);
    }
    public double nanmax(){
        double max = Double.MIN_VALUE;
        for(int i = 0; i<values.length;i++){
            for(int j = 0; j<dimension;j++){
              if(!Double.isNaN(values[i][j])&&values[i][j]>max){
                  max = values[i][j];
              }
            }
        }
        return  max == Double.MIN_VALUE ? Double.NaN : max;
    }
    public double nanmin(){
        double min = Double.MAX_VALUE;
        for(int i = 0; i<values.length;i++){
            for(int j = 0; j<dimension;j++){
                if(!Double.isNaN(values[i][j])&&values[i][j]<min){
                    min = values[i][j];
                }
            }
        }
        return  min == Double.MAX_VALUE ? Double.NaN : min;
    }
    public void fillnan(double value){
        for(int i = 0; i<values.length;i++){
            for(int j = 0; j<dimension;j++){
                if(Double.isNaN(values[i][j])){
                    values[i][j] = value;
                }
            }
        }
    }
    public RealMatrix toMatrix(){
        return MatrixUtils.createRealMatrix(ArrayUtils.transpose(values));
    }
    public RealMatrix toMatrix(int nRows, int nCols){
        if(dimension!= 1){
            throw new IllegalStateException("only applicable to univariat Time-series");
        }
        if(nRows*nCols!=values.length){
            throw new IllegalArgumentException("matrix must have ass many entries as the time-series");
        }
        return MatrixUtils.createRealMatrix(ArrayUtils.to2D(ArrayUtils.to1D(values),nRows,nCols));
    }
    public void setMissingValues(double value){
        for(int i = 0; i<length();i++){
            for(int j = 0; j<dimension;j++){
                if(Double.isNaN(values[i][j])) {
                    values[i][j] = value;
                }
            }
        }
    }
    public RealVector toVector(){
        if(dimension>1){
            throw new IllegalStateException("only supported for 1D timeseries");
        }
        return MatrixUtils.createRealVector(ArrayUtils.to1D(values));
    }
    public void plot(){
        plot("generic Titel");
    }
    public void plot(String titel){
        Plotter awt= new Plotter(titel,this);
        awt.pack();
        awt.setVisible(true);
    }
    public TimeSeries append(TimeSeries t){
        if(t.dimension!=dimension)
            throw new IlligelInput("dimensions of TS does not match");
        double[][] temp = new double[values.length+t.values.length][dimension];
        System.arraycopy(values,0,temp,0,values.length);
        System.arraycopy(t.values,0,temp,values.length,t.values.length);
        values = temp;
        return this;
    }
    public void setRow(double[] row, int index){
        if(row.length<length()){
            throw new IllegalArgumentException("input length is to smalll");
        }
        if(dimension<=index||index<0){
            throw new IllegalArgumentException("index is invalide");
        }
        for(int i = 0;i<length();i++){
            values[i][index] = row[i];
        }
    }

    public RealMatrix visibilityGraph(){
        if(dimension>1){
            throw new IllegalStateException("this operation is only supported for 1-D timeseries");
        }
        RealVector v = toVector();
        RealMatrix ret = MatrixUtils.createRealMatrix(v.getDimension(),v.getDimension());
        boolean intercept;
        double a;
        double b;
        for(int i = 0; i<v.getDimension();i++){
            for(int j = i+1;j<v.getDimension();j++){
                b = v.getEntry(i);
                a = (v.getEntry(j)-v.getEntry(i))/(j-i);
                intercept = false;
                for(int z = j-1; z>i;z--){
                    if(v.getEntry(z)>=a*(z-i)+b){
                        intercept=true;
                        break;
                    }
                }
                if(!intercept){
                    ret.setEntry(i,j,1);
                    ret.setEntry(j,i,1);
                }

            }
        }
        return ret;

    }
    public void rescale(){
        RealMatrix r = toMatrix();
        maxs = new double[r.getRowDimension()];
        mins = new double[r.getRowDimension()];
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                maxs[i] = ArrayUtils.max(r.getRowVector(i).toArray());
                mins[i] = ArrayUtils.nanMin(r.getRowVector(i).toArray());
                r.setRowVector(i,r.getRowVector(i).mapSubtract(mins[i]).mapDivide(maxs[i]));
            }
        }
        values = ArrayUtils.transpose(r.getData());
    }
    public void mean_shift() {
        RealMatrix r = toMatrix();
        mins = new double[r.getRowDimension()];
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                mins[i] = ArrayUtils.mean(r.getRowVector(i).toArray());
                r.setRowVector(i,r.getRowVector(i).mapSubtract(mins[i]));
            }
        }
        values = ArrayUtils.transpose(r.getData());
    }
    public void inverse_mean_shift(){
        RealMatrix r = toMatrix();
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapAdd(mins[i]));
            }
        }
        mins = null;
        values = ArrayUtils.transpose(r.getData());

    }
    public TimeSeries log_transform(){
        values = ArrayUtils.log(values);
        return this;
    }
    public RealMatrix inverse_mean_shift(RealMatrix r){
        RealMatrix ret = MatrixUtils.createRealMatrix(r.getRowDimension(),r.getColumnDimension());
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapAdd(mins[i]));
            }
        }
        return r;

    }
    public void shift(){
        RealMatrix r = toMatrix();
        mins = new double[r.getRowDimension()];
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                mins[i] = ArrayUtils.nanMin(r.getRowVector(i).toArray());
                r.setRowVector(i,r.getRowVector(i).mapSubtract(mins[i]));
            }
        }
        values = ArrayUtils.transpose(r.getData());
    }
    public void inverse_shift(){
        RealMatrix r = toMatrix();
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapAdd(mins[i]));
            }
        }
        mins = null;
        values = ArrayUtils.transpose(r.getData());

    }
    public RealMatrix inverse_shift(RealMatrix r){
        RealMatrix ret = MatrixUtils.createRealMatrix(r.getRowDimension(),r.getColumnDimension());
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapAdd(mins[i]));
            }
        }
        return r;

    }
    public RealMatrix inverse_scale(RealMatrix r){
        RealMatrix ret = MatrixUtils.createRealMatrix(r.getRowDimension(),r.getColumnDimension());
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapMultiply(maxs[i]).mapAdd(mins[i]));
            }
        }
        return r;

    }
    public void inverse_scale(){
        RealMatrix r = toMatrix();
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                r.setRowVector(i,r.getRowVector(i).mapMultiply(maxs[i]).mapAdd(mins[i]));
            }
        }
        maxs = null;
        mins = null;
        values = ArrayUtils.transpose(r.getData());


    }

    public String toString(){
        return Arrays.deepToString(values);
    }

    public void writeToCSV(String path) throws FileNotFoundException {
        try (PrintWriter writer = new PrintWriter(new File(path))) {
            StringBuilder sb = new StringBuilder();
            System.out.println(dimension);
            for(int i = 0; i<dimension;i++){
                for(int j = 0; j<length();j++){
                    sb.append(values[j][i]+(j == length()-1 ? "":","));
                }

                sb.append("\n");
            }
            writer.write(sb.toString());
            writer.flush();
        }
    }

    public TimeSeries getEnd(int len) {
        return getSubSeries(length()-len,len);
    }

    public double getEntry(int i, int j) {
        return values[j][i];
    }

    public TimeSeries inverseDiff(double[][] val2, boolean drop) {
        double[] last;
        for(double[] d : val2){
            last = d;
            double[][] temp = new double[values.length+1][];
            temp[0] = d;
            int count = 1;
            for(double[] cur : values){
                temp[count] = new double[cur.length];
                for(int j = 0; j<d.length;j++){
                    temp[count][j] = last[j]+cur[j];
                }
                count++;
                last = cur;
            }
            values = temp;
        }
        if(drop) {
            double[][] temp = new double[values.length-val2.length][dimension];
            System.arraycopy(values,val2.length,temp,0,values.length-val2.length);
            values = temp;
        }
        return this;
    }
    public int decay(){
        if(dimension>1){
            throw new Error("only supported for 1-D series");
        }
        RealMatrix temp = toMatrix();
        for(int i = 0;i<temp.getColumnDimension();i++){
            
        }
        return 0;
    }
    public TimeSeries inverseDiff() {
        RealMatrix ret = toMatrix();
        for(int i = 1; i<ret.getColumnDimension();i++){
            ret.setColumnVector(i,ret.getColumnVector(i-1).add(ret.getColumnVector(i)));
        }
        return new TimeSeries(ret);
    }
    public void plotComponentwise(){
        RealMatrix data = toMatrix();
        Graph[] graphList = new Graph[dimension];
        for(int i = 0; i<dimension;i++){
            graphList[i] = new Graph(new IndexSet("time",0,length()));
            graphList[i].add(new Integer(i).toString(),data.getRowMatrix(i));
            graphList[i].plot();

        }
    }


    public TimeSeries trailingMA(int windowSize) {
        RealMatrix data = this.toMatrix();
        RealMatrix result = MatrixUtils.createRealMatrix(dimension,length()-windowSize+1);
        for (int i = windowSize;i<=length();i++) {
            result.setColumnVector((i-windowSize),MatrixTools.sumColumn(data.getSubMatrix(0,data.getRowDimension()-1,(i-windowSize),(i-1))).mapMultiplyToSelf(windowSize));
        }
        return new TimeSeries(result);
    }

    public boolean containsNaN() {
        return true;
    }

    public boolean is1D() {
        return dimension == 1;
    }


    public TimeSeries fillMissingValuesConstantContinuation() {
        RealMatrix r = toMatrix();
        double last_seen = 0;
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                if(!Double.isNaN(r.getEntry(i,j))){
                    last_seen = r.getEntry(i,j);

                }
                else{
                    r.setEntry(i,j,last_seen);
                }
            }
            last_seen = 0;
        }
        return new TimeSeries(r);
    }

    public TimeSeries getNDSubseries(int start, int end) {
        RealMatrix temp = toMatrix();
        return new TimeSeries(temp.getSubMatrix(start,end,0,temp.getColumnDimension()-1));
    }


}
