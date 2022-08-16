import com.sun.xml.internal.bind.api.impl.NameConverter;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import thesis.Models.*;
import thesis.TSModel;
import thesis.Tools.Graph;
import thesis.Tools.TimeSeries;
import thesis.Tools.Tuple;


import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Aplication {
    private static Aplication instance;
    //list of scenario names
    protected ArrayList<String> scenarios;

    //each method is associated with with data sets, which intern are associated with a parameter name to value mapping
    //the method name is specified by its class name since reflection is used for initialization
    protected HashMap<String,HashMap<String,HashMap<String,String>>> hyper_params;
    //each data set is associated with a list of scenarios
    protected HashMap<String,ArrayList<String>> data_scenario_mapping;
    //method list
    protected ArrayList<String> methods;

    protected ArrayList<String> working_methods;
    protected ArrayList<String> working_datasets;


    public static Aplication getInstance(){
        if (instance==null)
            instance = new Aplication();

        return instance;
    }
    private Aplication(){
        load();
    }
    //check if name of dataseet is legal
    public void legal(String name){
        if(name.equalsIgnoreCase("ALL"))
            throw new Error("ALL is a reserved key word");
        if(name.contains("#"))
            throw new Error("# is not allowed to be contained in the name");
        if(name.contains("{"))
            throw new Error("{ is not allowed to be contained in the name");
        if(name.contains("}"))
            throw new Error("} is not allowed to be contained in the name");
        if(name.contains("]"))
            throw new Error("] is not allowed to be contained in the name");
        if(name.contains("["))
            throw new Error("[ is not allowed to be contained in the name");
        if(name.contains("="))
            throw new Error("= is not allowed to be contained in the name");
        if(name.contains("%"))
            throw new Error("% is not allowed to be contained in the name");
        if(name.contains("!"))
            throw new Error("! is not allowed to be contained in the name");
        if(name.contains(", "))
            throw new Error("char sequence \" ,\" is not allowed to be contained in the name");
        if(name.contains("***"))
            throw new Error("three consecutive * is not allowed to be contained in the name");
    }
    public void add_time_series(TimeSeries t, String name, String... Scenarios){
        legal(name);
        try {
            t.writeToCSV("Datasets\\"+name+".csv");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ArrayList<String> arr = new ArrayList<>();
        for(String s : Scenarios) {
            if(!scenarios.contains(s)){
                throw new Error("no such scenario: "+s);
            }
            arr.add(s);
        }
        data_scenario_mapping.put(name,arr);
        working_datasets.add(name);

    }
    public void add_time_series(RealMatrix t, String name, String... Scenarios){
        add_time_series(new TimeSeries(t),name,Scenarios);

    }
    public void add_time_series(String path, String name, String... Scenarios) throws IOException {
        add_time_series(new TimeSeries(path),name,Scenarios);
    }
    public void remove_time_series(String name){
        new File(path(name)).delete();
        //cascade delete in data structure
        ArrayList<String> to_remove = new ArrayList<>();
        data_scenario_mapping.forEach((k,v)->{
            //also delete all sub-series
            if(k.equalsIgnoreCase(name)||k.matches(name+"#"+"\\d+")){
                to_remove.add(k);
            }
        });
        to_remove.forEach(k->data_scenario_mapping.remove(k));
        to_remove.clear();
        hyper_params.forEach((k,v)->v.forEach((k2,v2)->{
            if(k2.equalsIgnoreCase(name)||k2.matches(name+"#"+"\\d+")){
                to_remove.add(k2);
            }
        }));
        to_remove.forEach(k->hyper_params.forEach((x,y)->y.remove(k)));
        working_datasets.remove(name);
    }
    public void add_subseries(String name, int index, String... Scenarios){
        if(!data_scenario_mapping.containsKey(name))
            throw new Error("no such time series");

        try {
            if(new TimeSeries(path(name)).getDimension()<=index)
                throw new Error("index is larger than the time series");
        } catch (IOException e) {
            e.printStackTrace();
        }

        name = name+"#"+new Integer(index).toString();
        ArrayList<String> arr = new ArrayList<>();
        for(String s : Scenarios) {
            if(!scenarios.contains(s)){
                throw new Error("no such scenario: "+s);
            }
            arr.add(s);
        }
        data_scenario_mapping.put(name,arr);
        working_datasets.add(name);

    }
    public void remove_subseries(String name, int index){
        String subname = subserie_name(name,index);

        //rove in hyperparams and mapping
        ArrayList<String> to_remove = new ArrayList<>();
        data_scenario_mapping.forEach((k,v)->{
            if(k.equalsIgnoreCase(subname)){
                to_remove.add(k);
            }
        });
        to_remove.forEach(k -> data_scenario_mapping.remove(k));
        to_remove.clear();

        hyper_params.forEach((k,v)->v.forEach((k2,v2)->{
            if(k2.equalsIgnoreCase(subname)){
                to_remove.add(k2);
            }
        }));
        to_remove.forEach(k->hyper_params.forEach((x,y)->y.remove(k)));
        working_datasets.remove(subname);

    }
    public void add_scenario(String name){
        legal(name);
        if(!scenarios.contains(name))
            scenarios.add(name);
    }
    public void remove_scenario(String name){
        legal(name);
        if(scenarios.contains(name))
            scenarios.remove(name);

        //cascade over mapping
        data_scenario_mapping.forEach((k,v)->{
            if(v.contains(name))
                v.remove(name);
        });
    }
    public void add_associated_scenario(String name, String... Scenarios){
        if(!data_scenario_mapping.containsKey(name))
            throw new Error("no such time series");

        ArrayList<String> cur = data_scenario_mapping.get(name);

        for(String s : Scenarios){
            if(!scenarios.contains(s)){
                throw new Error("no such scenario :"+s);
            }
            if(!cur.contains(s)){
                cur.add(s);
            }
        }
        data_scenario_mapping.put(name,cur);
    }
    public void remove_associated_scenario(String name, String... Scenarios){
        if(!data_scenario_mapping.containsKey(name))
            throw new Error("no such time series");

        ArrayList<String> cur = data_scenario_mapping.get(name);


        for(String s : Scenarios){
            if(!scenarios.contains(s)){
                throw new Error("no such scenario :"+s);
            }
            if(cur.contains(s)){
                cur.remove(s);
            }
        }
        data_scenario_mapping.put(name,cur);
    }
    //subseries are encoded by a # at the end of the data set name followed by the index
    public void add_associated_scenario(String name,int index, String... Scenarios){
        name = name+"#"+new Integer(index).toString();
        add_associated_scenario(name,Scenarios);
    }
    public void remove_associated_scenario(String name,int index, String... Scenarios){
        name = name+"#"+new Integer(index).toString();
        remove_associated_scenario(name,Scenarios);
    }

    public void add_method(TSModel method){
        String method_str = method.getClass().getName();
        if(!methods.contains(method_str))
            methods.add(method_str);
        working_methods.add(method_str);
    }
    public void remove_method(TSModel method){
        String method_str = method.getClass().getName();
        methods.remove(method_str);
        hyper_params.remove(method_str);
        working_methods.remove(method_str);
    }

    public void set_working_data_set(String name){
        working_datasets = new ArrayList<>();
        working_datasets.add(name);
        working_datasets.addAll(get_subseries(name));

    }

    private ArrayList<String> get_subseries(java.lang.String name) {
        ArrayList<String> returner = new ArrayList<>();
        data_scenario_mapping.forEach((k,v)-> {
            if(k.matches(name+"#"+"\\d+"))
                returner.add(k);
        });
        return returner;
    }

    public void add_working_method(TSModel ... methods){
        for(TSModel t : methods){
            if(!working_methods.contains(t))
                working_methods.add(t.getClass().getName());
        }
    }
    public void remove_working_method(TSModel ... methods){
        for(TSModel t : methods){
            working_methods.remove(t.getClass().getName());
        }
    }
    public void add_working_series(String ... series){
        for(String s : series){
            if(!working_datasets.contains(s))
                working_datasets.add(s);
        }
    }
    public void remove_working_series(String ... series){
        for(String s : series){
            working_datasets.remove(s);
        }
    }
    //add data set specific hyperparameter
    public void add_hyperparameters(TSModel method,String dataset, HashMap<String,String> Hyperparameters){
        String method_str = method.getClass().getName();
        if(!methods.contains(method_str))
            throw new Error("pls add method before calling this method");


        Hyperparameters.forEach((k,v)->{legal(k);legal(v);});
        if (!dataset.equalsIgnoreCase("ALL")&&!data_scenario_mapping.containsKey(dataset))
            throw new Error("Dataset is not defined");


        HashMap<String,String> cur = new HashMap();
        if(hyper_params.containsKey(method_str)&&hyper_params.get(method_str).containsKey(dataset))
            cur = hyper_params.get(method_str).get(dataset);

        cur.putAll(Hyperparameters);

        if(!hyper_params.containsKey(method_str)){
            hyper_params.put(method_str,new HashMap<>());
        }

        hyper_params.get(method_str).put(dataset,cur);

    }
    //remove data set specific hyperparameter
    public void remove_hyperparameters(TSModel method,String dataset, String... Hyperparameters){
        String method_str = method.getClass().getName();
        HashMap<String,String> cur = new HashMap();
        if (!dataset.equalsIgnoreCase("ALL")&&!data_scenario_mapping.containsKey(dataset))
            throw new Error("Dataset is not defined");

        if(hyper_params.containsKey(method_str)&&hyper_params.get(method_str).containsKey(dataset))
            cur = hyper_params.get(method_str).get(dataset);
        else
            return;

        for(String s : Hyperparameters) {
            cur.remove(s);
        }
        //remove dangling hashmap
        if(cur.isEmpty()){
            hyper_params.get(method_str).remove(dataset);
        }


    }

    public void add_hyperparameters(TSModel method, String dataset, int index, HashMap<String,String> Hyperparameters){
        dataset = dataset+"#"+new Integer(index).toString();
        add_hyperparameters(method,dataset,Hyperparameters);
    }

    public void remove_hyperparameters(TSModel method,String dataset,int index, String... Hyperparameters){
        dataset = dataset+"#"+new Integer(index).toString();
        remove_hyperparameters(method,dataset,Hyperparameters);
    }



    //Add global hyperparameters which are used if no specification is given for the particular data set
    public void add_hyperparameters(TSModel method, HashMap<String,String> Hyperparameters){
        add_hyperparameters(method,"ALL",Hyperparameters);
    }

    //remove global hyperparameters
    public void remove_hyperparameters(TSModel method, String... Hyperparameters){
        remove_hyperparameters(method,"ALL",Hyperparameters);
    }

    public void test_datasets(Test test){
        for(String method_str : working_methods){
            for(String data_set_str : working_datasets) {

                //first create the Hyperparameters Hahsmap
                HashMap<String,String> params = new HashMap<>();
                //if time series is a subseries store the name of the parent here. data sets are their own parents
                String parent_dataset_name = parent_data_set_name(data_set_str);
                if(hyper_params.containsKey(method_str)){
                    HashMap<String,HashMap<String,String>> cur = hyper_params.get(method_str);
                    //first add global settings
                    if(cur.containsKey("ALL"))
                        params.putAll(cur.get("ALL"));
                    if(cur.containsKey(parent_dataset_name))
                        params.putAll(cur.get(parent_dataset_name));
                    if(cur.containsKey(data_set_str))
                        params.putAll(cur.get(data_set_str));
                }

                try {
                    //check whether experiment need not be run since the output already exists
                    String dir_path = test.directory_path == null ? "Outputs" : test.directory_path;
                    String output_path = dir_path+"\\"+parent_dataset_name+(is_subseries(data_set_str) ? "\\"+index(data_set_str) : "")+"\\"+method_str+test.file_ending();

                    if(new File(output_path).exists()&&test.load_params(output_path).equals(params)){
                        continue;
                    }

                        //create instance of target object
                    Class<? extends TSModel> clazz = (Class<? extends TSModel>) Class.forName(method_str);
                    Constructor<? extends TSModel> ctor = clazz.getConstructor(HashMap.class);
                    TSModel model = ctor.newInstance(reformat(params));

                    //load dataset to make dimensionality check and decide how to continue
                    TimeSeries timeSeries = new TimeSeries(path(parent_dataset_name));
                    if(is_subseries(data_set_str)){
                        timeSeries = timeSeries.get1DSubSeries(index(data_set_str));
                    }
                    boolean ts_1_D = timeSeries.toMatrix().getRowDimension() == 1;


                    if(!ts_1_D&&!model.isND()||ts_1_D&&!model.is1D()){
                        continue;
                    }
                    System.out.println(data_set_str);
                    test.test(data_set_str,model,timeSeries,params,output_path);

                } catch (InstantiationException e) {
                    e.printStackTrace();
                    System.exit(-1);
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                    System.exit(-1);
                } catch (NoSuchMethodException e) {
                    e.printStackTrace();
                    System.exit(-1);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                    System.exit(-1);
                } catch (ClassNotFoundException e) {
                    e.printStackTrace();
                    System.exit(-1);
                } catch (IOException e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }

    }



    static ArrayList<String> decode_ArrayList(String encoding){
        if (encoding.equalsIgnoreCase("[]")||encoding.equalsIgnoreCase(""))
            return new ArrayList<>();
        encoding = encoding.replaceAll("\\[","");
        encoding = encoding.replaceAll("\\]","");
        String[] values = encoding.split(", ");
        ArrayList<String> ret = new ArrayList<>();
        ret.addAll(Arrays.asList(values));
        return ret;
    }
    private static HashMap<String,ArrayList<String>> decode_mapping(String encoding){
        if(encoding.equalsIgnoreCase("{}")||encoding.equalsIgnoreCase(""))
            return new HashMap<>();

        HashMap<String,ArrayList<String>> ret = new HashMap<>();
        encoding = encoding.replaceAll("\\{","");
        encoding = encoding.replaceAll("\\}","");
        char[] arr = encoding.toCharArray();
        StringBuilder key = new StringBuilder();
        StringBuilder list = new StringBuilder();
        int i = 0;
        while (i<arr.length) {
            while (arr[i] != '=')
                key.append(arr[i++]);
            i++;
            while (arr[i] != ']')
                list.append(arr[i++]);
            list.append(arr[i++]);
            ret.put(key.toString(), decode_ArrayList(list.toString()));
            i+= 2;
            key = new StringBuilder();
            list = new StringBuilder();
        }
        return ret;
    }
    private static HashMap<String,String> decode_hashmap(String encoding){
        if(encoding.equalsIgnoreCase("{}")||encoding.equalsIgnoreCase(""))
            return new HashMap<>();

        HashMap<String,String> returner = new HashMap<>();
        encoding = encoding.replaceAll("\\{","");
        encoding = encoding.replaceAll("\\}","");
        String[] split = encoding.split(", ");
        for(String s : split){
            String[] splitsplit = s.split("=");
            returner.put(splitsplit[0],splitsplit[1]);
         }
        return returner;
    }
    private HashMap<String,HashMap<String,HashMap<String,String>>> decode_hyperparams(String encoding){
        return new BracketTree(encoding).result;
    }

    private static class BracketTree{
        Node root = new Node();
        char[] encoding;
        HashMap<String,HashMap<String,HashMap<String,String>>> result;
        static class Node {
            Node parent;
            int start;
            int end;
            HashMap data;
            ArrayList<Node> children = new ArrayList<>();
        }
        BracketTree(String encoding){
            this.encoding = encoding.toCharArray();
            construct();
            construct_map(root);
            result = (HashMap<String,HashMap<String,HashMap<String,String>>>)root.children.get(0).data;

        }
        void print_tree(Node cur){
            for(int i = 0;i<cur.children.size();i++){
                print_tree(cur.children.get(i));
            }
            System.out.println("start: "+cur.start+" end: "+cur.end);
        }
        void construct_map(Node cur){
            for(int i = 0;i<cur.children.size();i++){
                construct_map(cur.children.get(i));
            }
            if(cur.children.size()==0){
                cur.data = decode_hashmap(new String(Arrays.copyOfRange(encoding,cur.start,cur.end+1)));
            }
            else {
                int i = 0;
                int j = cur.start+1;
                StringBuilder key = new StringBuilder();
                cur.data = new HashMap();
                while(j< cur.end){
                    while (encoding[j]!='=') {
                        key.append(encoding[j++]);
                    }
                    cur.data.put(key.toString(),cur.children.get(i).data);
                    j=cur.children.get(i++).end+3;
                    key = new StringBuilder();
                }
            }
        }
        private void construct() {
            Node cur = root;
            for (int i = 0; i < encoding.length; i++) {
                if (encoding[i] == '{') {
                    Node newNode = new Node();
                    newNode.start = i;
                    newNode.parent = cur;
                    cur.children.add(newNode);
                    cur = newNode;
                }
                else if (encoding[i] == '}') {
                    cur.end = i;
                    cur = cur.parent;
                }
            }
        }
    }


    //read this Object from file
    private void load(){
        try {
            List<String> encoding_list = Files.readAllLines(Paths.get("Settings.txt"));

            if(encoding_list.size()>1){
                throw new Error("something went terrible wrong with the settings");
            }
            if(encoding_list.size()==0||encoding_list.get(0).length()==0){
                scenarios = new ArrayList<>();
                hyper_params = new HashMap<>();
                data_scenario_mapping = new HashMap<>();
                methods = new ArrayList<>();
                working_methods = methods;
                working_datasets = new ArrayList<>(data_scenario_mapping.keySet());
                return;
            }
            String encoding = encoding_list.get(0);
            String[] split = encoding.split("\\*\\*\\*");
            scenarios = decode_ArrayList(split[0]);
            hyper_params = decode_hyperparams(split[1]);
            data_scenario_mapping = decode_mapping(split[2]);
            methods = decode_ArrayList(split[3]);
            working_methods = new ArrayList<>();
            for(String s : methods){
                working_methods.add(s);
            }
            working_datasets = new ArrayList<>(data_scenario_mapping.keySet());

        } catch (IOException e) {
            e.printStackTrace();
        }


    }
    //Write this Object to file
    private void save(){
        try {
            BufferedWriter b = new BufferedWriter(new FileWriter(new File("Settings.txt")));
            b.write(scenarios.toString());
            b.write("***");
            b.write(hyper_params.toString());
            b.write("***");
            b.write(data_scenario_mapping.toString());
            b.write("***");
            b.write(methods.toString());
            b.flush();
            b.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static HashMap<String,ArrayList<String>> reformat(HashMap<String, String> hyperParams) {
        HashMap<String,ArrayList<String>> ret = new HashMap<>();
        hyperParams.forEach((k,v) -> ret.put(k,new ArrayList<String>()));
        hyperParams.forEach((k,v) -> ret.get(k).add(v));
        return ret;

    }
    private static int index(String name){
        return  new Integer(name.substring(name.lastIndexOf("#") + 1).trim());


    }

    private static String path(String name) {
        return "Datasets\\"+name+".csv";
    }

    public static String subserie_name(String name, int index){
        return name+"#"+new Integer(index).toString();
    }
    private static String parent_data_set_name(String name){
        return name.replaceAll("#"+"\\d+","");
    }

    private static boolean is_subseries(String name){
        return name.substring(name.lastIndexOf("#")+1).matches("\\d+");
    }

    private static void store_output(TSModel.Output o,HashMap<String,String> params, String path) throws IOException {
        String code = o.toString()+"######"+params.toString();
        File f = new File(path);
        f.getParentFile().mkdirs();
        BufferedWriter writer = new BufferedWriter(new FileWriter(f));
        writer.write(code);
        writer.flush();

    }

    private static void store_vary_result(TSModel.Vary_Result v, HashMap<String, String> params, String output_path) throws IOException {
        String code = v.encode()+"##########"+params.toString();
        File f = new File(output_path);
        f.getParentFile().mkdirs();
        BufferedWriter writer = new BufferedWriter(new FileWriter(f));
        writer.write(code);
        writer.flush();

    }
    private static Tuple<HashMap<String,String>, TSModel.Vary_Result> load_vary_result(String path){
        File f = new File(path);
        TSModel.Vary_Result o = null;
        HashMap<String, String> h = null;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(f));
            String encoded = reader.lines().collect(Collectors.joining(System.lineSeparator()));
            String[] split = encoded.split("##########");
            o = TSModel.Vary_Result.decode_v_r(split[0]);
            h = decode_hashmap(split[1]);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return new Tuple<>(h,o);

    }

    private static Tuple<HashMap<String,String>, TSModel.Output> load_output(String path){
        File f = new File(path);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(f));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        String encoded = reader.lines().collect(Collectors.joining(System.lineSeparator()));
        String[] split = encoded.split("######");
        TSModel.Output o = TSModel.decode_output(split[0]);
        HashMap<String,String> h = decode_hashmap(split[1]);
        return new Tuple<>(h,o);
    }
    public static class Vary_Horizon extends Outputter{
        //the horizon is aggregated such that their are always 10 data points
        //dataset error graph mapping with truncation
        HashMap<String,Graph> trunc_error_graphs = new HashMap<>();
        //dataset error graph mapping without truncation
        HashMap<String,Graph> non_trunc_error_graphs = new HashMap<>();

        //Dataset maps to method-output mapping
        HashMap<String,HashMap<String, TSModel.Output>> aggregate = new HashMap<>();
        //dataset -> method -> runtime
        HashMap<String,HashMap<String,Double>> runtime_map = new HashMap<>();
        TSModel.Metric m;
        public Vary_Horizon(TSModel.Metric m){
            this.m = m;
        }

        public void plot_forecasts(String dataset, TSModel... methods){
            HashMap<String,TSModel.Output> temp =  aggregate.get(dataset);
            for (TSModel method : methods) {
                String description = method.getClass().getName();
                if (temp.containsKey(description)){
                    temp.get(description).plotComparrison(description,dataset);
                }
                description = method.getClass().getName()+"1D";
                if (temp.containsKey(description)){
                    temp.get(description).plotComparrison(description,dataset);
                }
                description = method.getClass().getName()+"ND";
                if (temp.containsKey(description)){
                    temp.get(description).plotComparrison(description,dataset);
                }
            }
        }

        public void plot_graphs(){
            non_trunc_error_graphs.forEach((d,g)-> {
                //g.set_log_base_y(2);
                g.plot();
            });
        }
        public String all_to_latex(HashMap<String,Graph> error_graphs){
            StringBuilder sb = new StringBuilder();
            error_graphs.forEach((k,v)->sb.append(to_latex_plot(v,k)));
            return sb.toString();

        }
        //todo:  change format to subplot format and add 1d and ND series to deffering plots:
        public String to_latex_plot(Graph g,String titel){
            StringBuilder sb = new StringBuilder();
            //PLOT HEADER
            sb.append("\\begin{subfigure}{\\textwidth}\n");
            sb.append("\\centering\n");
            sb.append("\\begin{tikzpicture}\n" +
                    "\\begin{axis}[\n" +
                    "\txlabel=Horizon,\n" +
                    "\tylabel="+m.getDescription()+","+
                    "\twidth=\\textwidth,height=.2\\textheight]\n");

            g.get_ys().forEach((k,v)-> {
                        sb.append("\\addplot[color=" + String_to_Color.get_color(k) + ",mark=x] coordinates {\n");
                        ArrayList<Double> indxs = g.get_indexs().indexs();
                        for (int i = 0; i < indxs.size(); i++) {
                            sb.append("\t(" + indxs.get(i) + "," + v.get(i) + ")\n");
                        }
                        sb.append("};\n");
                    });
            //Image closing tags
            sb.append("\\end{axis}\n");
            sb.append("\\end{tikzpicture}\n");
            sb.append("\\caption{"+titel.replaceAll("_"," ").replaceAll("#"," ")+"}\n");
            sb.append("\\end{subfigure}\n");

            return sb.toString();

        }
        //get color method assigns color to strings in a pre defined oreder of request kind of awkward if else due to 1D ND non ending resemblence of method
        private static class String_to_Color{
            private static final String[] colors = {"magenta","yellow","black","red","green","blue","cyan","lightgray","brown","lime","olive","orange","pink","purple","teal","violet"};
            private static HashMap<String,String> string_color_association = new HashMap<>();
            private static int count = 0;
            private static String get_color(String key){
                if(string_color_association.containsKey(key)){
                    return string_color_association.get(key);
                }
                if(key.substring(key.length()-2).equals("1D")&&string_color_association.containsKey(key.substring(0,key.length()-2)+"ND")){
                    return string_color_association.get(key.substring(0,key.length()-2)+"ND")+"!70!white";
                }
                if(key.substring(key.length()-2).equals("ND")&&string_color_association.containsKey(key.substring(0,key.length()-2)+"1D")){
                    return string_color_association.get(key.substring(0,key.length()-2)+"1D")+"!70!white";
                }
                if(string_color_association.containsKey(key+"1D")){
                    return string_color_association.get(key.substring(0,key.length()-2)+"1D")+"!70!white";
                }
                if(string_color_association.containsKey(key+"ND")){
                    return string_color_association.get(key+"ND");
                }
                if(string_color_association.containsKey(key.substring(0,key.length()-2))&&key.substring(key.length()-2).equals("ND")){
                    return string_color_association.get(key.substring(0,key.length()-2));
                }
                if(string_color_association.containsKey(key.substring(0,key.length()-2))&&key.substring(key.length()-2).equals("1D")){
                    return string_color_association.get(key.substring(0,key.length()-2))+"!70!white";
                }
                if(count==colors.length){
                    throw new Error("no further colors available");
                }
                string_color_association.put(key,colors[count++]);
                return string_color_association.get(key);
            }
        }

        @Override
        protected void conclude() {
            //remove 1D and ND tags incase of no duplicates
            for(String key : aggregate.keySet()){
                //add to seen when seen once and remove again when seen twice therefore only methods seen 1 time are kept
                ArrayList<String> seen = new ArrayList<>();
                ArrayList<String> keys = new ArrayList<>();
                for(String method : aggregate.get(key).keySet()){
                    String method_name = method.substring(0,method.length()-2);
                    if(!seen.contains(method_name)) {
                        seen.add(method_name);
                        keys.add(method);
                    }
                    else {
                        seen.remove(method_name);
                        keys.remove(method_name+"ND");
                        keys.remove(method_name+"1D");
                    }

                }
                for(String name : keys){
                    String method_name = name.substring(0,name.length()-2);
                    aggregate.get(key).put(method_name,aggregate.get(key).get(name));
                    aggregate.get(key).remove(name);
                }
            }
            aggregate.forEach((data_name,v)->
                    v.forEach((method_name,output)->{
                        int step_size = (int)Math.ceil(((double) output.getTest().getColumnDimension())/10);
                        Graph cur_trunc = Graph.all_horizon_graph_aggregated(data_name,method_name,output,m,step_size,true);
                        Graph cur_non_trunc = Graph.all_horizon_graph_aggregated(data_name,method_name,output,m,step_size,false);
                        if(!trunc_error_graphs.containsKey(data_name)){
                            trunc_error_graphs.put(data_name,cur_trunc);
                            non_trunc_error_graphs.put(data_name,cur_non_trunc);
                        }
                        else{
                           trunc_error_graphs.put(data_name,trunc_error_graphs.get(data_name).merge(cur_trunc));
                           non_trunc_error_graphs.put(data_name,non_trunc_error_graphs.get(data_name).merge(cur_non_trunc));
                        }
                    }));

        }

        @Override
        protected void process_dataset(String data_set_name, ArrayList<Integer> sub_series, File f) {
            String method_name = method_name(f)+"ND";
            if(!aggregate.containsKey(data_set_name))
                aggregate.put(data_set_name,new HashMap<>());

            aggregate.get(data_set_name).put(method_name,load_output(f.getPath()).getVal2());

            for(Integer index : sub_series) {
                String s_name = subserie_name(data_set_name,index);
                if(!aggregate.containsKey(s_name))
                    aggregate.put(s_name,new HashMap<>());

                aggregate.get(s_name).put(method_name,load_output(f.getPath()).getVal2().sub_copy(index));
            }

        }

        @Override
        protected void process_subseries(String data_set_name, int index, File f) {
            String method_name = method_name(f)+"1D";
            String s_name = subserie_name(data_set_name,index);
            if(!aggregate.containsKey(s_name))
                aggregate.put(s_name,new HashMap<>());

            aggregate.get(s_name).put(method_name,load_output(f.getPath()).getVal2());



        }

        @Override
        protected boolean relevant(String file_name) {
            return file_name.substring(file_name.lastIndexOf("_")+1).equalsIgnoreCase("standard.txt");
        }

    }
    public static class Vary_Length_Plotter extends Outputter{
        //dataset-> errorGraph
        HashMap<String,Graph> error_graphs = new HashMap<>();
        //dataset -> runtimegraph
        HashMap<String,Graph> runtime_Graphs = new HashMap<>();

        //Dataset maps to method-output mapping
        HashMap<String,HashMap<String, TSModel.Vary_Result>> aggregate = new HashMap<>();
        TSModel.Metric m;

        public Vary_Length_Plotter(TSModel.Metric m){
            this.m = m;
        }

        @Override
        protected void conclude() {
            //remove 1D and ND tags incase of no duplicates
            for(String key : aggregate.keySet()){
                //add to seen when seen once and remove again when seen twice therefore only methods seen 1 time are kept
                ArrayList<String> seen = new ArrayList<>();
                ArrayList<String> keys = new ArrayList<>();
                for(String method : aggregate.get(key).keySet()){
                    String method_name = method.substring(0,method.length()-2);
                    if(!seen.contains(method_name)) {
                        seen.add(method_name);
                        keys.add(method);
                    }
                    else {
                        seen.remove(method_name);
                        keys.remove(method_name+"ND");
                        keys.remove(method_name+"1D");
                    }

                }
                for(String name : keys){
                    String method_name = name.substring(0,name.length()-2);
                    aggregate.get(key).put(method_name,aggregate.get(key).get(name));
                    aggregate.get(key).remove(name);

                }
            }
            aggregate.forEach((data_name,v)->v.forEach((method,res)->{
                if(!error_graphs.containsKey(data_name)){
                    error_graphs.put(data_name,res.error_graph(m,data_name));
                    runtime_Graphs.put(data_name,res.runtime_graph(data_name));

                }
                else{
                    error_graphs.put(data_name,error_graphs.get(data_name).merge(res.error_graph(m,data_name)));
                    runtime_Graphs.put(data_name,runtime_Graphs.get(data_name).merge(res.runtime_graph(data_name)));
                }

            }));

        }

        public void plot_error_graphs(){
            error_graphs.forEach((d,g)-> {
                //g.set_log_base_y(2);
                g.plot();
            });
        }
        public void plot_runtime_graphs(){
            runtime_Graphs.forEach((d,g)-> {
                //g.set_log_base_y(2);
                g.plot();
            });
        }

        @Override
        protected void process_dataset(String data_set_name, ArrayList<Integer> sub_series, File f) {
            String method_name = method_name(f)+"ND";
            if(!aggregate.containsKey(data_set_name))
                aggregate.put(data_set_name,new HashMap<>());

            aggregate.get(data_set_name).put(method_name,load_vary_result(f.getPath()).getVal2());

            for(Integer index : sub_series) {
                String s_name = subserie_name(data_set_name,index);
                if(!aggregate.containsKey(s_name))
                    aggregate.put(s_name,new HashMap<>());

                aggregate.get(s_name).put(method_name,load_vary_result(f.getPath()).getVal2().subseries(index));
            }

        }

        @Override
        protected void process_subseries(String data_set_name, int index, File f) {
            String method_name = method_name(f)+"1D";
            String s_name = subserie_name(data_set_name,index);
            if(!aggregate.containsKey(s_name))
                aggregate.put(s_name,new HashMap<>());


            aggregate.get(s_name).put(method_name,load_vary_result(f.getPath()).getVal2());

        }

        @Override
        protected boolean relevant(String file_name) {
            return file_name.substring(file_name.lastIndexOf("_")+1).equalsIgnoreCase("length.txt");
        }
    }
    public class Scenario_Ranker extends Outputter{
        //scenario maps to hashmap which maps methods to list of errors encountered on data sets of that scenario.
        HashMap<String,HashMap<String,ArrayList<Double>>> scens = new HashMap<>();
        TSModel.Metric metric;
        public Scenario_Ranker(TSModel.Metric m){
            metric = m;
        }

        @Override
        protected void conclude() {
            //remove ND and 1D tags first of all if their is no ambiguity
            for(String key : scens.keySet()){
                //add to seen when seen once and remove again when seen twice therefore only methods seen 1 time are kept
                ArrayList<String> seen = new ArrayList<>();
                ArrayList<String> keys = new ArrayList<>();
                for(String method : scens.get(key).keySet()){
                    String method_name = method.substring(0,method.length()-2);
                    if(!seen.contains(method_name)) {
                        seen.add(method_name);
                        keys.add(method);
                    }
                    else {
                        seen.remove(method_name);
                        keys.remove(method_name+"ND");
                        keys.remove(method_name+"1D");
                    }

                }
                for(String name : keys){
                    String method_name = name.substring(0,name.length()-2);
                    scens.get(key).put(method_name,scens.get(key).get(name));
                    scens.get(key).remove(name);
                }
            }

            //now take the average to gain one number per method and scenario
            HashMap<String,HashMap<String,Double>> intermediate = new HashMap<>();
            scens.forEach((scenario,map)->scens.get(scenario).forEach((method,errors)->{
                if(!intermediate.containsKey(scenario))
                    intermediate.put(scenario,new HashMap<>());

                intermediate.get(scenario).put(method,new Mean().evaluate(ArrayUtils.toPrimitive(errors.toArray(new Double[0]))));
            }));

            display(intermediate);

        }
        private void display(HashMap<String,HashMap<String,Double>> error_mapping){
           HashMap<String,ArrayList<Tuple<String,Double>>> ranking = new HashMap<>();

           error_mapping.forEach((scenario, map)->
           {
               ranking.put(scenario,new ArrayList<>());
               ArrayList<Double> errors =  new ArrayList<>(map.values());
               Collections.sort(errors);
               ArrayList<String> exclude = new ArrayList<>();
               Double last = null;
               String last_name = null;
               for(Double error : errors){
                   if(last!= null && last.doubleValue() == error)
                       exclude.add(last_name);
                   else
                       exclude = new ArrayList<>();

                   last_name = getKeyByValue(error_mapping.get(scenario),error,exclude);
                   last = error;

                   ranking.get(scenario).add(new Tuple<>(last_name,error));

               }
           });

           ranking.forEach((scenario, rank)->{
               System.out.println("*****************************"+scenario+"*****************************");
               rank.forEach(pair -> System.out.println(pair.getVal1()+"            "+"Error: "+pair.getVal2()));
           });
        }

        @Override
        protected void process_dataset(String data_set_name, ArrayList<Integer> sub_series, File f) {
            String method_name = method_name(f)+"ND";
            TSModel.Output o = null;
            o = load_output(f.getPath()).getVal2();
            //Add errors to sub-scenarios
            for(Integer index : sub_series){
                String sub_name = subserie_name(data_set_name,index);
                ArrayList<String> cur_scen = data_scenario_mapping.get(sub_name);
                TSModel.Output sub_o = o.sub_copy(index);
                for(String scenario : cur_scen){
                    if(!scens.containsKey(scenario))
                        scens.put(scenario,new HashMap<>());

                    if(!scens.get(scenario).containsKey(method_name))
                        scens.get(scenario).put(method_name,new ArrayList<>());

                    scens.get(scenario).get(method_name).add(sub_o.error(metric));

                }

            }
            //add error to meain-scenarios
            for(String scenario : data_scenario_mapping.get(data_set_name)){
                if(!scens.containsKey(scenario))
                    scens.put(scenario,new HashMap<>());

                if(!scens.get(scenario).containsKey(method_name))
                    scens.get(scenario).put(method_name,new ArrayList<>());

                scens.get(scenario).get(method_name).add(o.error(metric));
            }


        }

        @Override
        protected void process_subseries(String data_set_name, int index, File f) {
            String method_name = method_name(f)+"1D";
            TSModel.Output o = null;
            o = load_output(f.getPath()).getVal2();
            //Add errors to sub-scenarios
            String sub_name = subserie_name(data_set_name,index);
            ArrayList<String> cur_scen = data_scenario_mapping.get(sub_name);
            for(String scenario : cur_scen){
                if(!scens.containsKey(scenario))
                    scens.put(scenario,new HashMap<>());

                    if(!scens.get(scenario).containsKey(method_name))
                        scens.get(scenario).put(method_name,new ArrayList<>());

                    scens.get(scenario).get(method_name).add(o.error(metric));

            }

        }




        @Override
        protected boolean relevant(String file_name) {
            return file_name.substring(file_name.lastIndexOf("_")+1).equalsIgnoreCase("standard.txt");
        }
        public<T, E> T getKeyByValue(Map<T, E> map, E value, ArrayList<T> exclude) {
            for (Map.Entry<T, E> entry : map.entrySet()) {
                if (Objects.equals(value, entry.getValue())&&!exclude.contains(entry.getKey())) {
                    return entry.getKey();
                }
            }
            return null;
        }
    }
    public static abstract class Outputter {
        //return name of method assuming the file follows the naming convention
        public String method_name(File f){
            return f.getName().substring(0,f.getName().lastIndexOf("_"));
        }
        void run(){
            for(File f : new File("Outputs").listFiles()){
                String data_set_name = f.getName();
                ArrayList<Integer> sub_series = new ArrayList<>();
                for(File f2 : f.listFiles()){
                    //all files in the directory are sub-series
                    if(f2.isDirectory()){
                        sub_series.add(new Integer(f2.getName()));
                    }
                }
                for(File f2 : f.listFiles()){
                    //all files in the directory are sub-series
                    if(f2.isDirectory()){
                        int index = new Integer(f2.getName());
                        for(File f3 : f2.listFiles()){
                            if(relevant(f3.getName()))
                                process_subseries(data_set_name,index,f3);
                        }
                    }
                    else{
                        if(relevant(f2.getName()))
                            process_dataset(data_set_name,sub_series,f2);
                    }
                }
            }
            conclude();
        }
        protected abstract void conclude();
        protected abstract void process_dataset(String data_set_name, ArrayList<Integer> sub_series, File f2);

        protected abstract void process_subseries(String data_set_name, int index, File f3);

        protected abstract boolean relevant(String file_name);

    }

    public static void main(String[] args) throws IOException {
        Aplication app = Aplication.getInstance();

        Vary_Horizon v = new Vary_Horizon(new TSModel.RMSE());
        v.run();
        //plotting all the error graphs
        v.plot_graphs();

        //example of how to plot an individual prediction
        //v.plot_forecasts("electricity_consumption#268",new trmf(),new TBATS());
        
        
        //uncomment this to plot the results of the vary length experiment (time series electricity_consumption#245: blackout series
        //excluded from analysis due to anomaly in test set)
        /*Vary_Length_Plotter v = new Vary_Length_Plotter(new TSModel.RMSE());
        v.run();
        //plotting all the error graphs
        v.plot_error_graphs();*/

        app.save();





    }
    public static void scale_comparison(){
        int[] subseries = {109,110};
        File f = new File("norm_false\\ATM_withdraw");
        String norm_true_path;
        for(File f1 : f.listFiles()){
            if(f1.isDirectory()){
                for(File f2 : f1.listFiles()){
                    norm_true_path = f2.getPath().replaceAll("norm_false","norm_true");
                    TSModel.Output not_norm = load_output(f2.getPath()).getVal2().normalize();
                    TSModel.Output normed = load_output(norm_true_path).getVal2();
                    System.out.println("Method: "+f2.getName()+"index:"+f.getParentFile().getName()+" non_norm :"+ not_norm.error(new TSModel.MAE())+" normed: "+normed.error(new TSModel.MAE()));

                }
            }
        }

    }

    public static abstract class Test {
        protected  String directory_path;
        abstract void test(String daset_name, TSModel method, TimeSeries timeSeries,HashMap<String,String> params, String output_path) throws IOException;
        abstract String file_ending();

        public abstract HashMap<String,String> load_params(String output_path) throws IOException;
    }
    public static class Standard_Test extends Test{
        Standard_Test(){

        }
        Standard_Test(String directory){
            directory_path = directory;
        }
        @Override
        void test(String daset_name,TSModel method, TimeSeries timeSeries,HashMap<String,String> params,String output_path) throws IOException {
            TSModel.Output o = method.test_accuracy(timeSeries.toMatrix(),0.8);
            store_output(o,params,output_path);

        }
        String file_ending(){
            return "_standard.txt";
        }
        public HashMap<String,String> load_params(String output_path) throws IOException {
            return load_output(output_path).getVal1();
        }
    }
    public static class Vary_Dimension extends Test{
        @Override
        void test(String daset_name,TSModel method, TimeSeries timeSeries, HashMap<String, String> params, String output_path) throws IOException {
            if(timeSeries.toMatrix().getColumnDimension()==1)
                return;
            TSModel.Vary_Result v = method.vary_dimension(timeSeries,daset_name);
            store_vary_result(v,params,output_path);
        }
        @Override
        String file_ending() {
            return "_vary_dimension.txt";
        }
        public HashMap<String,String> load_params(String output_path) throws IOException {
            return load_vary_result(output_path).getVal1();
        }
    }

    public static class Vary_Length extends Test{
        @Override
        void test(String daset_name,TSModel method, TimeSeries timeSeries, HashMap<String, String> params, String output_path) throws IOException {
            TSModel.Vary_Result v = method.vary_length(daset_name,timeSeries,0.1,0.1);
            store_vary_result(v,params,output_path);
        }
        @Override
        String file_ending() {
            return "_vary_length.txt";
        }
        public HashMap<String,String> load_params(String output_path) throws IOException {
            return load_vary_result(output_path).getVal1();
        }
    }
}
