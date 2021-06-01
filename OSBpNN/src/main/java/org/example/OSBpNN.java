package org.example;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.util.Random;

public class OSBpNN extends JPanel {
    double d1 = 0.1, d2 = 1, d3 = 1, d4 = 0, d5;
    double[][] dm1, dm2;
    double[][][] dm3, dm4, dm5;
    int[] im1, im2;
    int i1, i2, i3;
    long l1;
    Chart c;

    public OSBpNN(int[] im1, int[] im2) {
        im1[0] += 1;
        this.im1 = im1;
        this.im2 = im2;
        dm1 = new double[im1.length][];
        dm2 = new double[im1.length - 1][];
        dm4 = new double[im1.length - 1][][];
        dm3 = new double[im1.length - 1][][];
        dm5 = new double[im1.length - 1][][];
        for (int i = 0; i < im1.length; i++) {
            dm1[i] = new double[im1[i]];
        }
        dm1[0][dm1[0].length - 1] = 1;
        Random random = new Random();
        for (int i = 0; i < dm4.length; i++) {
            dm4[i] = new double[im1[i + 1]][im1[i] + 1];
            dm3[i] = new double[im1[i + 1]][im1[i] + 1];
            dm5[i] = new double[im1[i + 1]][im1[i] + 1];
            for (int j = 0; j < dm4[i].length; j++)
                for (int k = 0; k < dm4[i][j].length; k++) {
                    dm4[i][j][k] = random.nextDouble();
                }
        }
    }

    public double[] calc(double[] input) {
        if (dm1[0].length - 1 >= 0) System.arraycopy(input, 0, dm1[0], 0, dm1[0].length - 1);
        for (int i = 1; i < dm1.length; i++)
            for (int j = 0; j < dm1[i].length; j++) {
                double z = -dm4[i - 1][j][dm4[i - 1][j].length - 1];
                for (int k = 0; k < dm1[i - 1].length; k++) z += dm4[i - 1][j][k] * dm1[i - 1][k];
                dm1[i][j] = excit(z, im2[i - 1]);
            }
        return dm1[dm1.length - 1];
    }

    private double excit(double value, int icon) {
        switch (icon) {
            case 1 -> {
                return 2 / (1 + Math.exp(-2 * value)) - 1;
            }
            case 2 -> {
                return value;
            }
            default -> {
                return 1 / (1 + Math.exp(-value));
            }
        }
    }

    public double update(double[] target, double rate, double factor) {
        dm2[dm2.length - 1] = new double[target.length];
        for (int i = 0; i < dm2[dm3.length - 1].length; i++)
            dm2[dm2.length - 1][i] = deri(dm1[dm1.length - 1][i], im2[dm2.length - 1]) * (target[i] - dm1[dm1.length - 1][i]);
        for (int i = dm3.length - 2; i >= 0; i--) {
            dm2[i] = new double[dm1[i + 1].length];
            for (int j = 0; j < dm3[i].length; j++) {
                dm2[i][j] = 0;
                for (int k = 0; k < dm1[i + 2].length; k++) {
                    dm2[i][j] += dm2[i + 1][k] * dm4[i + 1][k][j];
                }
                dm2[i][j] *= deri(dm1[i + 1][j], im2[i]);
            }
        }
        for (int i = 0; i < dm3.length; i++)
            for (int j = 0; j < dm3[i].length; j++) {
                dm3[i][j][dm3[i][j].length - 1] = dm2[i][j] * rate;
                for (int k = 0; k < dm3[i][j].length - 1; k++)
                    dm3[i][j][k] = rate * dm2[i][j] * dm1[i][k] + factor * dm5[i][j][k];
            }
        for (int i = 0; i < dm3.length; i++)
            for (int j = 0; j < dm3[i].length; j++)
                System.arraycopy(dm3[i][j], 0, dm5[i][j], 0, dm3[i][j].length);
        revise();
        double sum = 0;
        for (int i = 0; i < target.length; i++) sum += 0.5 * Math.pow(target[i] - dm1[dm1.length - 1][i], 2);
        return sum;
    }

    private void revise() {
        for (int i = 0; i < dm4.length; i++) {
            for (int j = 0; j < dm4[i].length; j++) {
                for (int k = 0; k < dm4[i][j].length; k++) {
                    dm4[i][j][k] += dm3[i][j][k];
                }
            }
        }
    }

    private double deri(double element, int icon) {
        switch (icon) {
            case 1 -> {
                return Method.dtansig(element);
            }
            case 2 -> {
                return 1;
            }
            default -> {
                return Method.dsigmoid(element);
            }
        }
    }

    public void train(double[][] trainInput, double[][] trainTarget, double[][] valiInput, double[][] valiTarget, int iteration, double learnRate, double momentumRate, int icon, double lambda) {
        l1 = System.currentTimeMillis();
        this.d5 = lambda;
        i1 = trainInput.length;
        i2 = valiInput.length;
        JFrame frame = new JFrame("train sequence " + icon + " λ " + lambda + " size : " + trainInput.length);
        frame.setBounds(560, 240, 800, 600);
        c = new Chart();
        frame.add(this.c);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        double fundLearnRate = learnRate;
        for (i3 = 0; i3 < iteration; i3++) {
            learnRate = ((d3 - d1)) / d1 * fundLearnRate;
            d2 = 0;
            for (int j = 0; j < trainInput.length; j++) {
                double[] input = new double[trainInput[j].length];
                double[] target = new double[trainTarget[j].length];
                System.arraycopy(trainInput[j], 0, input, 0, trainInput[j].length);
                System.arraycopy(trainTarget[j], 0, target, 0, trainTarget[j].length);
                calc(input);
                d2 += update(target, learnRate, momentumRate);
            }
            d2 = Math.sqrt(d2 / trainInput.length);
            dr(valiInput, valiTarget);
            if (i3 % 10 == 0) {
                //System.out.println("train error = " + trainError);
                c.setTrainError(d2);
                c.repaint();
            }
        }
    }

    public void test(double[][] inputBatch, double[][] outputBatch) {
        StringBuilder sb = new StringBuilder();
        StringBuilder stringBuilder = new StringBuilder();
        long endTime = System.currentTimeMillis() - l1;
        stringBuilder.append("time elapse ").append(endTime / 1000).append(" s").append("\r\n").
                append("round : ").append(i3).append("\r\n").
                append("\r\n" + "spontaneous train rmse = ").append(d2).append("\r\n").
                append("min validation rmse = ").append(d4).append("\r\n").
                append("spontaneous validation rmse =").
                append(d3).append("\r\n").
                append("size T\\V:").append(i1).
                append(" \\ ").append(i2).
                append("\r\n").append("Structure of NN : ");
        for (int value : im1)
            stringBuilder.append(value).append(" ");
        stringBuilder.append("\r\n" + "activeFunction : ");
        for (int value : im2) stringBuilder.append(value).append(" ");
        stringBuilder.append("\r\n");
        stringBuilder.append("\r\n").append("parameter" + "\r\n");
        for (double[][] doubles : dm4)
            for (double[] aDouble : doubles)
                for (double v : aDouble)
                    stringBuilder.append(v).append("\r\n");
        try {
            d4 = 10;
            FileWriter fw = new FileWriter("_parameter" + ".out");
            fw.write("lambda : " + d5 + "\r\n");
            fw.write(stringBuilder.toString());
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < inputBatch.length; i++) {
            sb.append(" ").append(calc(inputBatch[i])[0]).append(" ").append(outputBatch[i][0]);
            sb.append("\r\n");
        }
        try {
            FileWriter fw = new FileWriter("data" + ".out");
            fw.write(sb.toString());
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void dr(double[][] inputBatch, double[][] targetBatch) {
        d3 = 0;
        for (int i = 0; i < inputBatch.length; i++) {
            calc(inputBatch[i]);
            for (int j = 0; j < targetBatch[0].length; j++)
                d3 += 0.5 * Math.pow(targetBatch[i][j] - dm1[dm1.length - 1][j], 2);
        }
        d3 = Math.sqrt(d3 / inputBatch.length);
        if (d4 != 0) {
            if (d3 < d4) {
                test(inputBatch, targetBatch);
                d4 = d3;
            }
        } else d4 = d3;
        c.setTestError(d3);
    }

    static class Method {

        public static double dsigmoid(double y) {
            return y * (1 - y);
        }

        public static double dtansig(double y) {
            return 1 - Math.pow(y, 2);
        }
    }

    static class Chart extends JPanel {
        int count = 0;
        int size = 100000;
        int[][] trainPairs;
        int[][] testPairs;
        int ymax;
        int ymin;
        private double divide = 1;
        private double trainError;
        private double testError;

        public Chart() {
            trainPairs = new int[size][2];
            testPairs = new int[size][2];
            setBounds(0, 0, 600, 800);
            trainPairs[0][0] = 0;
            trainPairs[0][1] = 1;
            trainPairs[1][0] = 100;
            trainPairs[1][1] = 1;
            ymax = 550;
            ymin = 0;
        }

        public void setTrainError(double trainError) {
            this.trainError = trainError;
        }

        public void setTestError(double testError) {
            this.testError = testError;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawLine(50, 500, 800, 500);
            g.drawLine(50, 0, 50, 500);
            g.setColor(new Color(234, 0, 0));
            for (int i = 0; i < count; i++) {
                if (trainPairs[i][1] > ymax) {
                    ymax = trainPairs[i][1];
                } else if (trainPairs[i][1] < ymin) {
                    ymin = trainPairs[i][1];
                }
                if (testPairs[i][1] > ymax) {
                    ymax = testPairs[i][1];
                } else if (testPairs[i][1] < ymin) {
                    ymin = testPairs[i][1];
                }
                int yDevide = 550 / (ymax - ymin);
                if (yDevide <= 1) {
                    yDevide = 2;
                }
                g.setColor(Color.blue);
                g.drawLine((int) ((trainPairs[i][0]) / (divide / 7.5)) + 50,
                        -trainPairs[i][1] * (yDevide - 1) + 500, (int) ((trainPairs[i + 1][0]) / (divide / 7.5)) + 50, -trainPairs[i + 1][1] * (yDevide - 1) + 500);
                g.setColor(Color.red);
                g.drawLine((int) ((testPairs[i][0]) / (divide / 7.5)) + 50,
                        -testPairs[i][1] * (yDevide - 1) + 500, (int) ((testPairs[i + 1][0]) / (divide / 7.5)) + 50, -testPairs[i + 1][1] * (yDevide - 1) + 500);
            }
            divide++;
            trainPairs[count + 2][0] = (count + 2) * 100;
            trainPairs[count + 2][1] = (int) (600 * trainError);
            testPairs[count + 2][0] = (count + 2) * 100;
            testPairs[count + 2][1] = (int) (600 * testError);
            count++;
            g.setColor(Color.black);
        }

    }

    public static void activate(int size, int[] layerNum, int[] activeFunction, int icon,double[][] array) {
        double[][] inputBatch = new double[array.length][9];
        double[][] outputBatch = new double[array.length][1];
        for (int i = 0; i < array.length; i++) {
            for (int j = 1; j < array[0].length; j++) {
                inputBatch[i][j - 1] = array[i][j];
            }
            outputBatch[i][0] = array[i][0];
        }
        int iteration = 100;
        double[][] trainInput = new double[size][inputBatch[0].length];
        double[][] trainTarget = new double[size][outputBatch[0].length];
        for (int i = 0; i < trainInput.length; i++) {
            System.arraycopy(inputBatch[i], 0, trainInput[i], 0, trainInput[0].length);
            trainTarget[i][0] = outputBatch[i][0];
        }
        int valiInputLength = 200;
        double[][] valiInput = new double[valiInputLength][inputBatch[0].length];
        double[][] valiTarget = new double[valiInputLength][outputBatch[0].length];
        for (int i = 0; i < valiInput.length; i++) {
            System.arraycopy(inputBatch[i + size], 0, valiInput[i], 0, valiInput[0].length);
            valiTarget[i][0] = outputBatch[i + size][0];
        }
        OSBpNN bpNNFunction = new OSBpNN(layerNum, activeFunction);
        bpNNFunction.train(trainInput, trainTarget, valiInput, valiTarget, iteration, 0.0000002, 0, icon, 0);
        bpNNFunction.test(trainInput, trainTarget);
    }
}
