package thesis.Tools;

import thesis.TSModel;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

public class Result_Manager {
    public static class All_Horizon_Error_Graphs{
        private File dir;
        private HashMap<String, HashMap<String, Graph>> error_graphs;
        public All_Horizon_Error_Graphs(String project_name,double split){
            try {
                dir =  new File("Results\\"+project_name);
                error_graphs = all_horizon_Result_Manager(project_name,split);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        public Graph get_graph(String data_set, String error_metric){
            return error_graphs.get(data_set).get(error_metric);
        }
        public void write_to_results(){

        }

    }
    public static HashMap<String, HashMap<String, Graph>> all_horizon_Result_Manager(String project_name, double split) throws IOException {
        HashMap<String,HashMap<String,HashMap<Integer,Graph>>> all_graphs = new HashMap<>();
        File zeProjectName = new File("Results\\"+project_name);
        for(File dataSet : zeProjectName.listFiles()){
            HashMap<String,HashMap<Integer,Graph>> fixed_dataSet = new HashMap<>();
            for(File model : dataSet.listFiles()){
                HashMap<Integer, Graph> fixed_model = new HashMap<>();
                for(File graphEncoding : model.listFiles()){
                    if(graphEncoding.toString().contains("varyHorizon")){
                        fixed_model.put(extract_integer(graphEncoding.toString()),Graph.read_from_file(graphEncoding.getPath()));

                    }
                    else
                        continue;
                }
                fixed_dataSet.put(model.toString(),fixed_model);
            }
            all_graphs.put(dataSet.toString(),fixed_dataSet);
        }
        HashMap<String,HashMap<String,Graph>>   Error_graphs = new HashMap<>();
        all_graphs.forEach((dataSet,graphs) -> Error_graphs.put(dataSet,construct_error_graphs(graphs,split,dataSet)));
        return Error_graphs;
    }

    private static HashMap<String, Graph> construct_error_graphs(HashMap<String, HashMap<Integer, Graph>> graphs, double split, String data_set) {
        HashMap<String,Graph> returner = new HashMap<>();
        for(TSModel.Metric m : TSModel.Metric.metrics()){
            returner.put(m.toString(),construct_error_graph(graphs,m, split,data_set));
        }
        return returner;
    }

    private static Graph construct_error_graph(HashMap<String, HashMap<Integer, Graph>> graphs, TSModel.Metric m, double split, String dat_set) {
        HashMap<String, HashMap<Integer, TSModel.Output>> outputs = new HashMap<>();
        graphs.forEach((model, all_horizons) -> {
            HashMap<Integer, TSModel.Output> fixed_model = new HashMap<>();
            all_horizons.forEach((horizon, graph) -> fixed_model.put(horizon,graph.to_output(split,TSModel.String_to_model(model, new HashMap<>()))));
            outputs.put(model,fixed_model);
        });
        HashMap<String, TSModel.MultiOutput> multi_outputs = to_multi_output(outputs);
        Graph error_graph = new Graph();
        multi_outputs.forEach((model,all_horizon)->Graph.merge(error_graph,all_horizon.error_graph(m,dat_set)));
        return error_graph;

    }

    private static HashMap<String, TSModel.MultiOutput> to_multi_output(HashMap<String, HashMap<Integer, TSModel.Output>> outputs) {
        HashMap<String, TSModel.MultiOutput> multi_outputs = new HashMap<>();
        outputs.forEach((model,outputs2)->multi_outputs.put(model,TSModel.String_to_model(model, new HashMap<>()).new MultiOutput(outputs2)));
        return multi_outputs;
    }

    private static Integer extract_integer(String string) {
        StringBuilder integer_part = new StringBuilder();
        for(char c : string.toCharArray()){
            if(c == '.'){
                break;
            }
            if(Character.isDigit(c)){
                integer_part.append(c);
            }
        }
        if(integer_part.length()==0){
            throw new Error("String does not contain any digits");
        }
        return new Integer(integer_part.toString());
    }
}
