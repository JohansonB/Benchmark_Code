package thesis.Plotting;
//code copied and adjusted from: https://www.tutorialspoint.com/jfreechart/jfreechart_timeseries_chart.htm
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.general.SeriesException;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.XYDataset;
import thesis.Models.*;
import thesis.TSModel;
import thesis.Tools.Graph;
import thesis.Tools.GridSearcher;


import java.io.IOException;
import java.util.HashMap;

public class Plotter extends ApplicationFrame {

    private JFreeChart chart;

    public Plotter(final String title , thesis.Tools.TimeSeries t) {
        super( title );
        final XYDataset dataset = createDataset(t);
        chart = createTSChart( dataset,"seconds","value" );
        final ChartPanel chartPanel = new ChartPanel( chart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 370 ) );
        chartPanel.setMouseZoomable( true , false );
        setContentPane( chartPanel );
    }
    public Plotter(final String title , XYDataset dataset, String indexsString, String errorString, boolean withShapes) {
        super( title);
        chart = createLineChart( dataset,indexsString,errorString,withShapes,false,10,false,2);


        final ChartPanel chartPanel = new ChartPanel( chart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 370 ) );
        chartPanel.setMouseZoomable( true , false );
        setContentPane( chartPanel );
    }
    public Plotter(final String title , XYDataset dataset, String indexsString, String errorString, boolean withShapes,boolean log_scale_y, int y_base,boolean log_scale_x, int x_base ) {
        super( title);
        chart = createLineChart( dataset,indexsString,errorString,withShapes,log_scale_y,y_base,log_scale_x,x_base);

        final ChartPanel chartPanel = new ChartPanel( chart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 370 ) );
        chartPanel.setMouseZoomable( true , false );
        setContentPane( chartPanel );
    }
    public JFreeChart getChart(){
        return chart;
    }

    private XYDataset createDataset(thesis.Tools.TimeSeries t) {
        TimeSeriesCollection colec = new TimeSeriesCollection();
        TimeSeries series;
        Second current;
        for(int j = 0; j<t.getDimension();j++) {
            current = new Second( );
            series = new TimeSeries( j );
            for (int i = 0; i < t.length(); i++) {

                try {
                    series.add(current, t.getEntry(j,i));
                    current = (Second) current.next();
                } catch (SeriesException e) {
                    System.err.println("Error adding to series");
                }
            }
            colec.addSeries(series);
        }

        return colec;
    }

    private JFreeChart createTSChart( final XYDataset dataset, String index, String value ) {
        JFreeChart chart =  ChartFactory.createTimeSeriesChart(
                "",
                index,
                value,
                dataset,
                false,
                false,
                false);


        return chart;
    }
    private JFreeChart createLineChart(XYDataset dataset, String index, String value,boolean withShapes,boolean log_scale_y,int y_base,boolean log_scale_x,int x_base){
        JFreeChart chart =  ChartFactory.createXYLineChart("",index,value,dataset);
        chart.removeLegend();
        if(log_scale_y){
            XYPlot plot = chart.getXYPlot();
            LogAxis yAxis = new LogAxis(value);
            yAxis.setBase(y_base);
            yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
            plot.setRangeAxis(yAxis);
        }
        if(log_scale_x){
            XYPlot plot = chart.getXYPlot();
            LogAxis xAxis = new LogAxis(index);
            xAxis.setBase(x_base);
            xAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
            plot.setDomainAxis(xAxis);
        }
        if(withShapes) {
            XYPlot plot = chart.getXYPlot();
            XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
            java.awt.geom.Ellipse2D.Double shape = new java.awt.geom.Ellipse2D.Double(-2.0, -2.0, 4.0, 4.0);
            renderer.setDefaultShape(shape);
            renderer.setDefaultShapesVisible(true);
        }
        return chart;
    }


}