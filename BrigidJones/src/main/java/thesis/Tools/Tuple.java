package thesis.Tools;

public class Tuple<K,V> {
    K val1;
    V val2;
    public Tuple(K val1, V val2){
        this.val1 = val1;
        this.val2 = val2;
    }
    public K getVal1(){
        return val1;
    }
    public V getVal2(){
        return val2;
    }
    @Override
    public boolean equals(Object o){
        if(o==null||!(o instanceof Tuple)){
            return false;
        }
        if(((Tuple) o).val1.equals(val1)&&((Tuple) o).val2.equals(val2)){
            return true;
        }
        return false;
    }
    public String toString(){
        return val1.toString()+" "+val2.toString();
    }

}
