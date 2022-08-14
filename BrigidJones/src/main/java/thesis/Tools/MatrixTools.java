package thesis.Tools;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class MatrixTools {

    public static RealMatrix divideComponentwise(RealMatrix nomi, RealMatrix deno) {
        RealMatrix C = MatrixUtils.createRealMatrix(nomi.getRowDimension(),nomi.getColumnDimension());
        for(int i = 0; i<C.getRowDimension();i++){
            for(int j = 0; j<C.getColumnDimension();j++){
                C.setEntry(i,j,nomi.getEntry(i,j)/deno.getEntry(i,j));
            }
        }
        return C;
    }

    public static RealMatrix pow(RealMatrix r, double v) {
        RealMatrix ret = MatrixUtils.createRealMatrix(r.getRowDimension(),r.getColumnDimension());
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                ret.setEntry(i,j,Math.pow(r.getEntry(i,j),v));
            }
        }
        return ret;
    }

    public static RealMatrix getUpper(RealMatrix matrix, int box, int slot) {
        RealMatrix ret = matrix.copy();
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        for(int i = 0; i<rows;i++){
            for(int j = 0; j<cols;j++){
                if(i*cols+j>box*cols+slot){
                   ret.setEntry(i,j,Double.NaN);
                }
            }
        }
        return ret;
    }
    public static RealMatrix getLower(RealMatrix matrix, int box, int slot) {
        RealMatrix ret = matrix.copy();
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        for(int i = 0; i<rows;i++){
            for(int j = 0; j<cols;j++){
                if(i*cols+j<=box*cols+slot){
                    ret.setEntry(i,j,Double.NaN);
                }
            }
        }
        return ret;
    }

    public static void fill(RealMatrix target, RealMatrix src, int startRow, int startCol, int endRow, int endCol) {
        for(int i = startRow;i<target.getRowDimension();i++){

            for(int j = startCol;j<target.getColumnDimension();j++){
                if(i == endRow||j==endCol){
                    break;
                }
                target.setEntry(i,j,src.getEntry(i,j));
            }
        }
    }

    public static Tuple<RealMatrix, RealMatrix> split(RealMatrix ts, double split) {
        int col = (int)(split*ts.getColumnDimension());
        RealMatrix train = ts.getSubMatrix(0,ts.getRowDimension()-1,0,col-1);
        RealMatrix test = ts.getSubMatrix(0,ts.getRowDimension()-1,col,ts.getColumnDimension()-1);
        return new Tuple<>(train,test);
    }

    public static RealVector randomVector(int len, int i, int i1) {
        UniformRealDistribution x = new UniformRealDistribution(-1,1);
        RealVector ret = MatrixUtils.createRealVector(new double[len]);
        for(int index = 0; index<len;index++){
            ret.setEntry(index,x.sample());
        }
        return ret;
    }

    public static double missRatio(RealMatrix train) {
        int count = 0;
        for(int i = 0; i<train.getRowDimension();i++){
            for(int j = 0; j<train.getColumnDimension();j++){
                if(!Double.isNaN(train.getEntry(i,j)))
                    count++;
            }
        }
        return ((double)count)/(train.getColumnDimension()*train.getRowDimension());
    }

    public static void setRange(RealMatrix data, int i0, int i1,double val) {
        for(int i = i0; i<= i1;i++){
            for(int j = 0; j<data.getRowDimension();j++){
                data.setEntry(j,i,val);
            }
        }

    }

    public static ArrayList<RealMatrix> subMatrices(RealMatrix data, int len) {
        int count = 0;
        ArrayList<RealMatrix> ret = new ArrayList<>();
        while((count+1)*len<=data.getColumnDimension())
            ret.add(data.getSubMatrix(0,data.getRowDimension()-1,count*len,++count*len-1));

        return ret;
    }

    public static RealMatrix covarianveToCorrelation(RealMatrix covarianceMatrix) {
        RealMatrix ret = MatrixUtils.createRealMatrix(covarianceMatrix.getRowDimension(),covarianceMatrix.getColumnDimension());
        for(int i = 0; i<covarianceMatrix.getRowDimension();i++){
            for(int j = 0; j<covarianceMatrix.getColumnDimension();j++){
                ret.setEntry(i,j,covarianceMatrix.getEntry(i,j)/Math.sqrt(covarianceMatrix.getEntry(i,i)*covarianceMatrix.getEntry(j,j)));
            }
        }
        return ret;
    }
    public static RealMatrix subMatrix(RealMatrix r, ArrayList<Integer> rowIndexes){
        RealMatrix ret = MatrixUtils.createRealMatrix(rowIndexes.size(),r.getColumnDimension());
        for(int i = 0; i<rowIndexes.size();i++){
            ret.setRowVector(i,r.getRowVector(rowIndexes.get(i)));
        }
        return ret;
    }

    public static RealMatrix columnMerge(RealMatrix A, RealMatrix B) {
        if(A == null){
            return B;
        }
        if(B== null){
            return A;
        }
        if(A.getRowDimension()!=B.getRowDimension()){
            throw new Error("row dimension of both matrecis must match");
        }
        RealMatrix ret = MatrixUtils.createRealMatrix(A.getRowDimension(),A.getColumnDimension()+B.getColumnDimension());
        ret.setSubMatrix(A.getData(),0,0);
        ret.setSubMatrix(B.getData(),0,A.getColumnDimension());
        return ret;
    }
    public static RealMatrix rowMerge(RealMatrix A, RealMatrix B){
        if(A == null){
            return B;
        }
        if(B== null){
            return A;
        }
        if(A.getColumnDimension() !=B.getColumnDimension()){
            throw new Error("column dimension of both matrecis must match");
        }
        RealMatrix ret = MatrixUtils.createRealMatrix(A.getRowDimension()+B.getRowDimension(),A.getColumnDimension());
        ret.setSubMatrix(A.getData(),0,0);
        ret.setSubMatrix(B.getData(),A.getRowDimension(),0);
        return ret;
    }

    public static RealMatrix infinityToNan(RealMatrix forecast) {
        RealMatrix ret = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
        for(int i = 0; i<forecast.getRowDimension();i++){
            for (int j = 0; j<forecast.getColumnDimension();j++){
                ret.setEntry(i,j,Double.isInfinite(forecast.getEntry(i,j)) ? Double.NaN : forecast.getEntry(i,j));
            }
        }
        return ret;
    }

    public static boolean containsNaN(RealMatrix R) {
        for(int i = 0; i<R.getRowDimension();i++){
            for(int j = 0; j<R.getColumnDimension();j++){
                if(Double.isNaN(R.getEntry(i,j))){
                    System.out.println("i: "+i+" j: "+j);
                    return true;
                }
            }
        }
        return false;
    }

    public static RealMatrix indexedRowMerge(int index, RealMatrix ... mats) {
        int count = 0;
        HashMap<Double,ArrayList<Double>> elCounter = new HashMap<>();
        RealVector indexes;
        RealVector dataVector;
        for(RealMatrix data : mats){
            if(data.getRowDimension()!=2){
                throw new Error("matrices need to have row dimension 2");
            }
            indexes = data.getRowVector(index);
            dataVector = data.getRowVector(1-index);
            for(Double key : elCounter.keySet()){
                if(!contains(indexes,key)) {

                    elCounter.get(key).add(Double.NaN);
                }
            }
            for(int i = 0; i<dataVector.getDimension();i++){
                if(!elCounter.containsKey(indexes.getEntry(i))){
                    ArrayList<Double> temp = new ArrayList<>();
                    for(int j = 0; j<count;j++){
                        temp.add(Double.NaN);
                    }
                    elCounter.put(indexes.getEntry(i),temp);
                }
                elCounter.get(indexes.getEntry(i)).add(dataVector.getEntry(i));

            }
            count++;
        }
        SortedSet<Double> keys = new TreeSet<>(elCounter.keySet());
        RealMatrix ret = MatrixUtils.createRealMatrix(mats.length,keys.size());
        count = 0;
        for (Double key : keys) {
            ArrayList<Double> value = elCounter.get(key);
            ret.setColumn(count++,ArrayUtils.toPrimitive(value.stream().toArray(Double[]::new)));
        }
        return ret;
    }

    private static boolean contains(RealVector indexes, Double key) {
        for(int i = 0; i<indexes.getDimension();i++){
            if(indexes.getEntry(i)==key){
                return true;
            }
        }
        return false;
    }

    public static boolean allNaN(RealMatrix r) {
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                if(!new Double(r.getEntry(i,j)).equals(Double.NaN)){
                    return false;
                }
            }
        }
        return true;
    }

    public static double ratio(RealMatrix Matrix, double v) {
        int count = 0;
        for(int i = 0;i<Matrix.getRowDimension();i++){
            for(int j = 0; j<Matrix.getColumnDimension();j++){
                if(new Double(v).equals(Matrix.getEntry(i,j))){
                    count++;
                }
            }
        }
        return count/(double)(Matrix.getColumnDimension()*Matrix.getRowDimension());
    }

    public static RealMatrix replace(RealMatrix r, double from, double to) {
        for(int i = 0; i<r.getRowDimension();i++){
            for(int j = 0; j<r.getColumnDimension();j++){
                if(new Double(r.getEntry(i,j)).equals(from))
                    r.setEntry(i,j,to);
            }
        }
        return r;
    }

    public static RealMatrix head(RealMatrix test, int len) {
        RealMatrix returner = test.getSubMatrix(0,test.getRowDimension()-1,0,len-1);
        return returner;
    }

    public static ArrayList<Tuple<Integer, Integer>> NaNs(RealMatrix M) {
        ArrayList<Tuple<Integer,Integer>> nans = new ArrayList<>();
        for (int i = 0; i<M.getRowDimension();i++){
            for(int j = 0; j<M.getColumnDimension();j++){
                if(Double.isNaN(M.getEntry(i,j))) {
                    nans.add(new Tuple<>(i, j));
                }
            }
        }
        return nans;
    }

    public static RealMatrix hadamard_prduct(RealMatrix past_values, RealMatrix weights) {
        RealMatrix ret = MatrixUtils.createRealMatrix(past_values.getRowDimension(),past_values.getColumnDimension());
        for(int i = 0; i<past_values.getRowDimension();i++){
            for(int j = 0; j<past_values.getColumnDimension();j++){
                ret.setEntry(i,j,past_values.getEntry(i,j)*weights.getEntry(i,j));
            }
        }
        return ret;
    }

    public static RealMatrix reverse(RealMatrix weights) {
        RealMatrix ret = MatrixUtils.createRealMatrix(weights.getRowDimension(),weights.getColumnDimension());
        int count = 0;
        for(int i = weights.getColumnDimension()-1;i>=0;i--){
            ret.setColumn(count++,weights.getColumn(i));
        }
        return ret;
    }

    public static RealMatrix leftShift(RealMatrix r, int i) {
        return r.getSubMatrix(0,r.getRowDimension()-1,i,r.getColumnDimension()-1);
    }

    public static RealMatrix normalize(RealMatrix M) {
        RealMatrix ret = MatrixUtils.createRealMatrix(M.getRowDimension(),M.getColumnDimension());
        for(int i = 0; i<ret.getRowDimension();i++){
            RealVector cur = M.getRowVector(i);
            double min = Collections.min(Arrays.asList(ArrayUtils.toObject(cur.toArray())));
            double max = Collections.max(Arrays.asList(ArrayUtils.toObject(cur.toArray())));
            ret.setRowVector(i,cur.mapSubtract(min).mapDivide(max-min));
        }
        return ret;
    }
    public static ArrayList<RealMatrix> normalize(RealMatrix M, RealMatrix ... rest) {
        RealMatrix ret1 = M.copy();
        ArrayList<RealMatrix> ret = new ArrayList<>();
        ret.add(ret1);
        for(RealMatrix r : rest){
            ret.add(r.copy());
        }
        for(int i = 0; i<ret1.getRowDimension();i++){
            RealVector cur = M.getRowVector(i);
            double min = Collections.min(Arrays.asList(ArrayUtils.toObject(cur.toArray())));
            double max = Collections.max(Arrays.asList(ArrayUtils.toObject(cur.toArray())));
            for(RealMatrix r : ret) {
                cur = r.getRowVector(i);
                r. setRowVector(i, cur.mapSubtract(min).mapDivide(max - min));
            }
        }
        return ret;
    }


    public static class MatrixIterator implements Iterator<Double> {
        RealMatrix r;
        int i = 0;
        int n;


        int j=0;
        int m;


        MatrixIterator(RealMatrix in){
            r = in;
            n = in.getRowDimension();
            m = in.getColumnDimension();
        }
        @Override
        public boolean hasNext() {
            return !(i==n&&j==m);
        }

        @Override
        public Double next() {
            if(i==n){
                i=0;
                j++;
            }
            return r.getEntry(i++,j);
        }
    }
    public static String getDimensions(RealMatrix in){
        return "("+in.getRowDimension() + "," + in.getColumnDimension()+")";
    }
    public static double[] flatten(RealMatrix mat, String mode){
        double[] ret = new double[mat.getColumnDimension()*mat.getRowDimension()];
        double[][] data = mat.getData();
        int count = 0;
        if(mode.equalsIgnoreCase("F")){
            for(int j = 0;j<mat.getColumnDimension();j++){
                for(int i = 0; i<mat.getRowDimension();i++){
                    ret[count++] = data[i][j];
                }
            }
        }
        return ret;
    }
    public static double[] flatten(RealMatrix mat){
        double[] ret = new double[mat.getColumnDimension()*mat.getRowDimension()];
        double[][] data = mat.getData();
        int count = 0;

        for(int j = 0;j<mat.getColumnDimension();j++){
            for(int i = 0; i<mat.getRowDimension();i++){
                ret[count++] = data[i][j];
            }
        }

        return ret;
    }
    public static RealMatrix multiplyComponenWise(RealMatrix A,RealMatrix B){
        if(A.getColumnDimension()!=B.getColumnDimension()||A.getRowDimension()!= B.getRowDimension()){
            throw new IlligelInput("dimensions of matrices dont match");
        }
        RealMatrix C = MatrixUtils.createRealMatrix(A.getRowDimension(),A.getColumnDimension());
        for(int i = 0; i<A.getRowDimension();i++){
            for(int j = 0; j<A.getColumnDimension();j++){
                C.setEntry(i,j,A.getEntry(i,j)*B.getEntry(i,j));
            }
        }
        return C;
    }
    public static double mean(RealMatrix A){
        return ArrayUtils.mean(A.getData());
    }
    public static RealMatrix boundUpper(RealMatrix R,double max){
        for(int i = 0; i<R.getRowDimension();i++){
            for(int j = 0; j<R.getColumnDimension();j++){
                if(R.getEntry(i,j)>max){
                    R.setEntry(i,j,max);
                }
            }
        }
        return R;
    }
    public static RealMatrix boundLower(RealMatrix R,double min){
        for(int i = 0; i<R.getRowDimension();i++){
            for(int j = 0; j<R.getColumnDimension();j++){
                if(R.getEntry(i,j)<min){
                    R.setEntry(i,j,min);
                }
            }
        }
        return R;
    }
    public static RealMatrix selectColumns(RealMatrix R, RealVector v){
        //validity check
        for(int i = 0; i<v.getDimension();i++){
            if(((int)v.getEntry(i))!=v.getEntry(i)){
                throw new IllegalArgumentException("input ust contain only positive integer values");
            }
        }
        RealMatrix ret = MatrixUtils.createRealMatrix(R.getRowDimension(),v.getDimension());
        for(int i = 0; i<v.getDimension();i++){
            ret.setColumn(i,R.getColumn((int)v.getEntry(i)));
        }
        return ret;
    }
    public static RealMatrix selectColumns(RealMatrix R, ArrayList<Integer> v){
        RealMatrix ret = MatrixUtils.createRealMatrix(R.getRowDimension(),v.size());
        for(int i = 0; i<v.size();i++){
            ret.setColumn(i,R.getColumn(v.get(i)));
        }
        return ret;
    }
    public static RealMatrix selectColumns(RealMatrix R, int[] v){
        RealMatrix ret = MatrixUtils.createRealMatrix(R.getRowDimension(),v.length);
        for(int i = 0; i<v.length;i++){
            ret.setColumn(i,R.getColumn(v[i]));
        }
        return ret;
    }
    public static RealMatrix setColumns(RealMatrix R, RealVector v,RealMatrix M){
        if(M.getColumnDimension()!=v.getDimension()){
            throw new IllegalArgumentException("M's column dimension must be the same as the dimension of v");
        }
        for(int i = 0;i<v.getDimension();i++) {
            if (((int)v.getEntry(i))!=v.getEntry(i)){
                throw new IllegalArgumentException("input ust contain only positive integer values");
            }
        }
        for(int i = 0; i<v.getDimension();i++){
            R.setColumn((int)v.getEntry(i),R.getColumn(i));
        }
        return R;
    }
    public static RealMatrix setColumns(RealMatrix R, ArrayList<Integer> v,RealMatrix M){
        if(M.getColumnDimension()!=v.size()){
            throw new IllegalArgumentException("M's column dimension must be the same as the dimension of v");
        }
        for(int i = 0; i<v.size();i++){
            R.setColumn(v.get(i),M.getColumn(i));
        }
        return R;
    }

    public static RealMatrix round(RealMatrix in) {
        for(int i = 0; i<in.getRowDimension();i++){
            for(int j = 0; j<in.getColumnDimension();j++){
                in.setEntry(i,j,Math.round(in.getEntry(i,j)));
            }
        }
        return in;

    }

    public static void print(RealMatrix r) {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i<r.getRowDimension();i++){
            double[] cur = r.getRow(i);
            for(int j=0; j<cur.length;j++){
                sb.append(cur[j]+"  ");
            }
            sb.append("\n\n");
        }
        for(int i = 0; i<r.getColumnDimension();i++) {
            sb.append("_____");
        }
        System.out.println(sb.toString());
    }


    public static double[] sumAxis(RealMatrix r,int axis) {
        double[] ret;
            double[][] arr = axis == 0 ? r.getData() : r.transpose().getData();
            ret = ArrayUtils.zeros(arr[0].length);
            for(double[] d : arr){
                ArrayUtils.selfAdd(ret,d);
            }
    return ret;
    }
    public static RealVector sumColumn(RealMatrix r) {
        RealVector ret = MatrixUtils.createRealVector(new double[r.getRowDimension()]);
        for (int i = 0; i<r.getColumnDimension();i++){
            ret = ret.add(r.getColumnVector(i));
        }
        return ret;
    }
    public static void replace_nans(RealMatrix r, double val){
        for(int i = 0; i<r.getRowDimension();i++ ){
            for(int j = 0; j<r.getColumnDimension();j++){
                if(Double.isNaN(r.getEntry(i,j))){
                    r.setEntry(i,j,val);
                }
            }
        }
    }
    public static ArrayList<Tuple<Integer,Integer>> getNans(RealMatrix r){
        ArrayList<Tuple<Integer,Integer>> ret = new ArrayList<>();
        for(int i = 0; i<r.getRowDimension();i++ ){
            for(int j = 0; j<r.getColumnDimension();j++){
                if(Double.isNaN(r.getEntry(i,j))){
                   ret.add(new Tuple<>(i,j));
                }
            }
        }
        return ret;
    }
    public static RealMatrix setEntries(RealMatrix r,ArrayList<Tuple<Integer,Integer>> coordinates, double val){
        for(Tuple<Integer,Integer> t : coordinates){
            r.setEntry(t.getVal1(),t.getVal2(),val);
        }
        return r;
    }
    public static void setEntries(RealMatrix toUpdate,RealMatrix from, ArrayList<Tuple<Integer,Integer>> cor){
        for(Tuple<Integer,Integer> c : cor ){
            toUpdate.setEntry(c.getVal1(),c.getVal2(),from.getEntry(c.getVal1(),c.getVal2()));
        }
    }

    public static RealMatrix mask(ArrayList<Tuple<Integer, Integer>> mask, RealMatrix r) {
        for(Tuple<Integer,Integer> t : mask){
            r.setEntry(t.getVal1(),t.getVal2(),0);
        }
        return r;
    }

    public static RealMatrix repeatCollumnVector(RealVector columnVector, int t) {
        RealMatrix ret = MatrixUtils.createRealMatrix(columnVector.getDimension(),t);
        for(int i = 0; i<t;i++) {
            ret.setColumnVector(i,columnVector);
        }
        return ret;
    }

    public static RealMatrix mapMultiply(RealMatrix x, RealMatrix w_l) {
        RealMatrix ret = MatrixUtils.createRealMatrix(x.getRowDimension(),x.getColumnDimension());
        for(int i = 0; i<x.getColumnDimension();i++){
            ret.setColumnVector(i,x.getColumnVector(i).ebeMultiply(w_l.getColumnVector(i)));
        }
        return ret;
    }

    public static RealMatrix collumnRoll(RealMatrix x_l, int lag) {
        return lag>=0 ? collumnRollSub(x_l,lag) : collumnRollSub(x_l,x_l.getColumnDimension()+lag);

    }
    private static RealMatrix collumnRollSub(RealMatrix x_l, int lag){
        RealMatrix ret = MatrixUtils.createRealMatrix(x_l.getRowDimension(),x_l.getColumnDimension());

        for(int i = 0; i< x_l.getColumnDimension();i++){
            int a = (i+lag);
            int b = (x_l.getColumnDimension());
            int c = (i+lag)%(x_l.getColumnDimension());
            ret.setColumnVector((i+lag)%(x_l.getColumnDimension()),x_l.getColumnVector(i));
        }
        return ret;
    }
    public static RealVector scalarAdd(RealVector a, double b){
        double[] temp = new double[a.getDimension()];
        for(int i = 0; i<a.getDimension();i++){
            temp[i] = b;
        }
        RealVector ret = MatrixUtils.createRealVector(temp);
        return a.add(ret);
    }

    public static RealMatrix columnConcat(RealMatrix x, RealMatrix y) {
        RealMatrix d = MatrixUtils.createRealMatrix(new double[x.getRowDimension()][x.getColumnDimension()+y.getColumnDimension()]);
        for(int i = 0;i<x.getColumnDimension();i++){
            d.setColumnVector(i,x.getColumnVector(i));
        }
        for(int i = 0; i<y.getColumnDimension();i++){
            d.setColumnVector(i+x.getColumnDimension(),y.getColumnVector(i));
        }
        return d;
    }

    public static double sum(RealVector v) {
        double tot = 0;
        for(double d : v.toArray()){
            tot+= d;
        }
        return tot;
    }

    public static RealMatrix rowMult(RealVector v, RealMatrix r) {
        RealMatrix ret = MatrixUtils.createRealMatrix(r.getRowDimension(),r.getColumnDimension());
        for(int i = 0; i<r.getRowDimension();i++){
            ret.setRowVector(i,r.getRowVector(i).ebeMultiply(v));
        }
        return ret;
    }

    public static RealVector reverse(RealVector subVector) {
        return MatrixUtils.createRealVector(ArrayUtils.reverse(subVector.toArray()));
    }

    public static RealVector insert(RealVector v, double e, int i) {
        return MatrixUtils.createRealVector(ArrayUtils.insert(v.toArray(),e,i));
    }
    //power function  ([a,b,c],e) -> [a^e,b^e,c^e]
    //does not alter input vector
    public static RealVector pow(RealVector v, double e) {
        double[] temp = v.toArray();
        for(int i = 0; i<temp.length;i++){
            temp[i] = Math.pow(temp[i],e);
        }
        return MatrixUtils.createRealVector(temp);
    }

    public static RealVector getSubVector(RealVector v, int i, int i1) {
        int count = 0;
        double[] ret = new double[i1-i+1];
        double[] temp = v.toArray();
        for(int index = i ; index<=i1;index++){
            ret[count++] = temp[index];
        }
        return MatrixUtils.createRealVector(ret);
    }

    public static RealMatrix randomMatrix(int n, int m,double lowerBound, double upperBound) {
        RealMatrix ret = MatrixUtils.createRealMatrix(n,m);
        UniformRealDistribution x = new UniformRealDistribution(lowerBound,upperBound);
        for(int i = 0; i<n;i++){
            for(int j = 0; j<m;j++){
                ret.setEntry(i,j,x.sample());
            }
        }
        return ret;


    }
    public static RealMatrix augment(RealMatrix in,int dim){
        RealMatrix ret = MatrixUtils.createRealMatrix(in.getRowDimension()+dim,in.getColumnDimension()+dim);
        ret.setSubMatrix(in.getData(),0,0);
        return ret;
    }

}
