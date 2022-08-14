package thesis.Tools;

import java.util.ArrayList;
import java.util.Arrays;

public class IndexSet {
    private String description;
    private ArrayList<Double> indexs = new ArrayList<>();
    public IndexSet(String description){
        this.description = description;
    }
    public IndexSet(String description, ArrayList<Double> indexs){
        this.description = description;
        this.indexs = indexs;
    }
    public IndexSet(String description, int start, int end){
        this.description = description;
        for(int i = start;i<end;i++){
            indexs.add((double) i);
        }
    }

    public IndexSet(String horizon, int start, int end, int step_size) {
        this.description = description;
        for(int i = start;i<=end;i+=step_size){
            indexs.add((double) i);
        }
        if(end%step_size!=0){
            indexs.add((double)end);
        }
    }


    public String getDescription(){
        return description;
    }
    public ArrayList<Double> indexs(){
        return indexs;
    }
    public String toString(){
        return description +"\n"+
                Arrays.toString(indexs.stream().toArray(Double[]::new)).replaceAll("\\[","").replaceAll("\\]","");

    }
    public static IndexSet parse_indexset(String encoding){
        String[] temp = encoding.split("\n");
        IndexSet ret = new IndexSet(temp[0]);
        for(String s : temp[1].split(", ")) {
            ret.indexs.add(new Double(s));
        }
        return ret;

    }
}
