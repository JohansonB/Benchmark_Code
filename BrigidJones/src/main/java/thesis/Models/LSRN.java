package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import thesis.TSModel;
import thesis.Tools.Graph;
import thesis.Tools.MatrixTools;
import thesis.Tools.TimeSeries;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class LSRN extends TSModel {


    private String scriptPath = "Scripts\\C\\LSRN\\LSRNmain.exe";
    private String pathTots = "Scripts\\C\\Temp\\LSRN_in.csv";


    private double d;//dimension of latent space
    private int steps;
    private double gamma;
    private double lambda;
    private int len;

    private static int defaultD = 2;
    private static int defaultSteps = 3;
    private static double defaultGamma = 0.03215;
    private static double defaultLambda = 8;

    public LSRN(){
        super();
    }
    public LSRN(HashMap<String, ArrayList<String>> in) {
        super(in);
    }

    protected void parse(HashMap<String,ArrayList<String>> in){
        d = in.containsKey("-latentDim") ? new Double(in.get("-latentDim").get(0)) : defaultD;
        lambda = in.containsKey("-lambda2") ? new Double(in.get("-lambda2").get(0)) : defaultLambda;
        gamma = in.containsKey("-gamma2") ? new Double(in.get("-gamma2").get(0)) : defaultGamma;
        steps = in.containsKey("-steps") ? new Double(in.get("-steps").get(0)).intValue() : defaultSteps;



    }








    private void executeScript() {
        try {
            String s = scriptPath + " " +  //path to script
                    pathTots + " " +
                    steps + " " +
                    lambda + " " +
                    gamma + " " +
                    (int)d + " " +
                    len;
            // run the Unix "ps -ef" command
            // using the Runtime exec method:
            System.out.println(s);
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


        } catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }
    }

    protected Output evaluate(){
        ArrayList<RealMatrix> ws = createWs(o.getTrain());

        d = d<1 ? (int)(o.getTrain().getRowDimension()*d) : d;

        if(steps>o.getTrain().getColumnDimension()+1){
            steps = o.getTrain().getColumnDimension()+1;
        }
        TimeSeries temp = new TimeSeries(ts);
        RealMatrix tsOG = ts;
        temp.shift();
        ts = temp.toMatrix();
        o = new Output(ts,split);
        try {
            new TimeSeries(o.getTrain().transpose()).writeToCSV(this.pathTots);
            for(int i = 0; i<ws.size();i++) {
                new TimeSeries(ws.get(i)).writeToCSV("Scripts\\C\\Temp\\LSRN_w"+i+".csv");
            }
            len = o.getTest().getColumnDimension()/steps;
            len = o.getTest().getColumnDimension()%steps == 0 ? len : len+1;
            executeScript();
            readOutput(o);

        } catch (IOException e  ) {
            e.printStackTrace();
        }
        o.setTrain(temp.inverse_shift(o.getTrain()));
        o.setTest(temp.inverse_shift(o.getTest()));
        o.setForecast(temp.inverse_shift(o.getForecast()));
        ts = tsOG;
        return o;
    }

    private void readOutput(Output o) throws IOException {
        String path = "Scripts\\C\\Output\\LSRN_out"+0+".csv";
        File f = new File(path);
        int count = 0;
        int vecCount = 0;
        RealMatrix cur;
        RealVector v = MatrixUtils.createRealVector(new double[o.getTest().getColumnDimension()]);
        while(f.exists()){
            cur = new TimeSeries(path).toMatrix();
            path = "Scripts\\C\\Output\\LSRN_out"+(++count)+".csv";
            for(int j = 0; j<steps;j++){
                if(vecCount<v.getDimension())
                    v.setEntry(vecCount++,cur.getEntry(j,j+1));
            }
            f.delete();
            f = new File(path);
            o.setRunTime(new Double(new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\LSRN_runtime.txt"))).readLine()));

        }
        System.out.println();
        o.setForecast(MatrixUtils.createRowRealMatrix(v.getSubVector(0,o.getTest().getColumnDimension()).toArray()));
    }
    private RealMatrix readOutput(MultiOutput o) throws IOException {
        String path = "Scripts\\C\\Output\\LSRN_out"+0+".csv";
        File f = new File(path);
        int count = 0;
        int vecCount = 0;
        RealMatrix cur;
        RealVector v = MatrixUtils.createRealVector(new double[o.getTest().getColumnDimension()]);
        while(f.exists()){
            cur = new TimeSeries(path).toMatrix();
            path = "Scripts\\C\\Output\\LSRN_out"+(++count)+".csv";
            for(int j = 0; j<steps;j++){
                if(vecCount<v.getDimension())
                    v.setEntry(vecCount++,cur.getEntry(j,j+1));
            }
            f.delete();
            f = new File(path);
            o.setRunTime(new Double(new BufferedReader(new FileReader(new File("Scripts\\C\\Output\\LSRN_runtime.txt"))).readLine()));

        }
        return MatrixUtils.createRowRealMatrix(v.getSubVector(0,o.getTest().getColumnDimension()).toArray());
    }
    protected MultiOutput compute_all_horizons(MultiOutput out){
        ArrayList<RealMatrix> ws = createWs(out.getTrain());

        d = d<1 ? (int)(out.getTrain().getRowDimension()*d) : d;

        if(steps>out.getTrain().getColumnDimension()+1){
            steps = out.getTrain().getColumnDimension()+1;
        }

        try {
            new TimeSeries(out.getTrain().transpose()).writeToCSV(this.pathTots);
            for(int i = 0; i<ws.size();i++) {
                new TimeSeries(ws.get(i)).writeToCSV("Scripts\\C\\Temp\\LSRN_w"+i+".csv");
            }
            len = out.getTest().getColumnDimension()/steps;
            len = out.getTest().getColumnDimension()%steps == 0 ? len : len+1;
            executeScript();
            RealMatrix res = readOutput(out);
            for(int i = 1 ; i<=out.getTest().getColumnDimension();i++){
                out.add_horizon(i,res);
            }

        } catch (IOException e  ) {
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


    private ArrayList<RealMatrix> createWs(RealMatrix data){
        RealMatrix w = MatrixUtils.createRealMatrix(steps+1,steps+1);
        ArrayList<RealMatrix> ret = new ArrayList<>();
        ArrayList<RealMatrix>  splits = MatrixTools.subMatrices(data,steps);
        for(int i = 0; i<splits.size();i++){
            //ret.add(MatrixTools.augment(new TimeSeries(splits.get(i)).visibilityGraph(),1));
            ret.add(new TimeSeries(splits.get(i)).inverseDiff(new double[][]{{0}},false).visibilityGraph());

        }
        return ret;
    }
    public HashMap<String,String[]> getSearchSpace(){
        HashMap<String,String[]> space = new HashMap<>();
        space.put("-latentDim",new String[]{"0.2","0.4","0.7"});
        space.put("-lambda2",new String[]{"0.0078125","0.03125","0.125","0.5","2","8","32"});
        space.put("-gamma2",new String[]{"0.0078125","0.03125","0.125","0.5","2","8","32"});
        space.put("-steps",new String[]{"10","30","50","100","200","500"});
        return space;
    }

    public String toString(){
        return "LSRN";
    }

    private static int nextSquare(int in){
        return (int)Math.ceil(Math.sqrt(in));
    }
}