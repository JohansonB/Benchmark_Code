package thesis.Tools;

public class Timer {
    private long startTime;
    public void start(){
        startTime = System.nanoTime();
    }
    public double stop(){
        long temp = startTime;
        startTime = 0;
        long end = System.nanoTime();
        long dif = end - temp;

        return temp == 0 ? 0 : dif*1e-9;
    }
}
