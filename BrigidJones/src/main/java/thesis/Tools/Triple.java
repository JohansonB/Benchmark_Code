package thesis.Tools;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.SymbolAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;

import javax.swing.*;
import java.awt.*;

public class Triple<A,B,C> {
    A val1;
    B val2;
    C val3;
    public Triple(A a ,B b,C c){
        val1 = a;
        val2 = b;
        val3 = c;
    }
    public A getVal1(){
        return val1;
    }
    public B getVal2(){
        return val2;
    }
    public C getVal3(){
        return val3;
    }

    @Override
    public boolean equals(Object o){
        if(o==null||!(o instanceof Tuple)){
            return false;
        }
        if(((Triple) o).val1.equals(val1)&&((Triple) o).val2.equals(val2)&&val3.equals(((Triple) o).val3)){
            return true;
        }
        return false;
    }


}
