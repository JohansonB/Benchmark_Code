package thesis.Tools;

import org.apache.commons.math3.linear.RealMatrix;

import thesis.TSModel;

import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

public class GridSearcher {

    public static void optimize(TSModel tsModel, RealMatrix train, String toString) {
        GridSearcher r = new GridSearcher(tsModel,train,toString);
        tsModel.update_params(r.search_bootStrap());
    }

    public TSModel getBestInstance() {
        return bestInstance;
    }

    public static interface Gridsearchable{
        HashMap<String,String[]> getSearchSpace();
    }

    private static class Iterer{
        private static class CappedInt{
            int val;
            int cap;
            CappedInt(int cap){
                this.cap = cap;
                val = 0;
            }
            boolean isCap(){
                return val == cap;
            }
            void increment(){
                val = isCap() ? 0 : val+1;
            }

        }
        CappedInt[] encoding;
        Iterer(ArrayList<Integer> in){
            encoding = new CappedInt[in.size()];
            int count = 0;
            for(Integer i : in){
                encoding[count++] = new CappedInt(i);
            }
        }
        ArrayList<int[]> allCombis(){
            ArrayList<int[]> ret = new ArrayList();

            do {
                ret.add(getValue());
            }
             while(increment());

            return ret;

        }

        private boolean increment() {
            int count = 0;
            do{
                if(count==encoding.length){
                    return false;
                }
                encoding[count].increment();

            }while(encoding[count++].val==0);
            return true;
        }

        private int[] getValue() {
            int[] ret = new int[encoding.length];
            for(int i = 0; i<ret.length;i++){
                ret[i] = encoding[i].val;
            }
            return ret;
        }




    }
    Class<? extends TSModel> model;

    RealMatrix trainingData;
    HashMap<String, String[]> paramsSpace;
    ArrayList<HashMap<String,String>> openParams;
    private HashMap<String, String> bestParams;
    private TSModel bestInstance;
    private String dataName;
    private TSModel curInstance;
    public GridSearcher(TSModel instance, RealMatrix trainingData, String dataName){
        model = instance.getClass();
        paramsSpace = instance.getSearchSpace();
        this.trainingData = trainingData;
        initOpenParams();
        this.dataName = dataName;
    }

    private void initOpenParams() {
        ArrayList<Integer> in = new ArrayList<>();
        paramsSpace.forEach((k,v)->in.add(v.length-1));
        Iterer iterer = new Iterer(in);
        ArrayList<int[]> combinations = iterer.allCombis();
        openParams = new ArrayList<>();
        HashMap<String,String> cur;

        for(int[] indexs : combinations){
            cur = new HashMap<>();
            for(int i = 0; i<indexs.length;i++){
                String key = (String) paramsSpace.keySet().toArray()[i];
                String value = paramsSpace.get(key)[indexs[i]];
                cur.put(key,value);
            }
            openParams.add(cur);
        }
        Collections.shuffle(openParams);

    }

    //do random gridsearch on 100 hyperparameter settings and return the one with the lowest RMSE
    public HashMap<String,ArrayList<String>> search_crossValid() {
        try {
            //hyperParameter selection
            Stack<HashMap<String, String>> working_stack = new Stack<>();
            working_stack.addAll(openParams);

            double bestError = Double.MAX_VALUE;
            HashMap<String, String> bestParams = null;
            HashMap<String, String> hyperParams = null;
            String dataPath = "Datasets\\Temp\\temp.csv";
            Constructor<? extends TSModel> con = model.getConstructor(HashMap.class);
            double lengthFract = 0.4;
            double stepsize = 0.01;
            double curError;
            TSModel.Output curOutput;
            for (double i = lengthFract; i < 1 && !working_stack.isEmpty(); i += stepsize) {
                System.out.println(i);
                hyperParams = working_stack.pop();
                curInstance = con.newInstance(reformat(hyperParams));
                int startColumn = (int) ((i - lengthFract) * trainingData.getColumnDimension());
                int endColumn = (int) (i * trainingData.getColumnDimension());
                new TimeSeries(trainingData.getSubMatrix(0, trainingData.getRowDimension() - 1, startColumn, endColumn)).writeToCSV(dataPath);

                if (bestError > (curError = (curOutput=curInstance.test_accuracy(dataPath, 0.7)).error(new TSModel.RMSE()))) {
                    bestError = curError;
                    bestParams = hyperParams;
                    bestInstance = curInstance;
                }
                System.out.println(curError);



            }
            this.bestParams = bestParams;
            return reformat(bestParams);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;

    }

    public HashMap<String,ArrayList<String>> search_bootStrap() {

        int max_iter = 40;
        double blockFraction = 0.2;
        double validationSplit = 0.9;


        try {
            //hyperParameter selection
            Stack<HashMap<String, String>> working_stack = new Stack<>();
            working_stack.addAll(openParams);
            double bestError = Double.MAX_VALUE;
            HashMap<String, String> bestParams = null;
            HashMap<String, String> hyperParams = null;
            String dataPath = "Datasets\\Temp\\temp.csv";
            Constructor<? extends TSModel> con = model.getConstructor(HashMap.class);
            TSModel.Output curOutput;

            double curError;

            for (double i = 0; i < max_iter && !working_stack.isEmpty(); i++) {
                System.out.println(i);
                hyperParams = working_stack.pop();
                curInstance = con.newInstance(reformat(hyperParams));

                new TimeSeries(bootstrap(blockFraction)).writeToCSV(dataPath);

                try {
                    curOutput = curInstance.test_accuracy(dataPath, validationSplit);
                }
                catch(IllegalParameters e){
                    curOutput = null;
                }
                if (curOutput !=null && bestError > (curError = curOutput.error(new TSModel.RMSE()))) {
                    bestError = curError;
                    bestParams = hyperParams;
                    bestInstance = curInstance;
                    System.out.println(bestParams);
                        //curOutput.plotComparrison();
                }




            }
            this.bestParams = bestParams;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.getCause().printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return reformat(bestParams);

    }

    private RealMatrix bootstrap(double blockFraction) {
        RealMatrix ret = null;
        Random r = new Random();
        int number = (int)(1/blockFraction);
        int blockLen = (int)(trainingData.getColumnDimension()*blockFraction);
        int bound = trainingData.getColumnDimension()-blockLen;
        int index;
        for(int i = 0; i<number;i++){
            ret = MatrixTools.columnMerge(ret,trainingData.getSubMatrix(0,trainingData.getRowDimension()-1,(index=r.nextInt(bound)),index+blockLen));
        }
        return ret;
    }



    public static HashMap<String,ArrayList<String>> reformat(HashMap<String, String> hyperParams) {
        HashMap<String,ArrayList<String>> ret = new HashMap<>();
        hyperParams.forEach((k,v) -> ret.put(k,new ArrayList<String>()));
        hyperParams.forEach((k,v) -> ret.get(k).add(v));
        return ret;

    }
    public void save() throws IOException {
        BufferedWriter b;
        b = new BufferedWriter(new FileWriter(new File("ModelConfigurations\\"+dataName+"_"+curInstance.toString()+".txt")));
        b.write(bestParams.toString());
        b.close();

    }



}
