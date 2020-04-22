package com.yolo3test;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

//import com.streambase.com.gs.collections.impl.Counter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;


public class yolo {

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }
    public static void main1(String[] args) throws InterruptedException {
        System.load("F:\\work-space\\hot_yolov3\\src\\data\\opencv_java411.dll");
        String modelWeights = "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3_18000.weights";
        String modelConfiguration = "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3.cfg";
        String filePath = "D:\\cars.mp4";
        VideoCapture cap = new VideoCapture(filePath);
        Mat frame = new Mat();
        Mat dst = new Mat ();
        //cap.read(frame);
        JFrame jframe = new JFrame("Video");
        JLabel vidpanel = new JLabel();
        jframe.setContentPane(vidpanel);
        jframe.setSize(600, 600);
        jframe.setVisible(true);

        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        //Thread.sleep(5000);

        //Mat image = Imgcodecs.imread("D:\\yolo-object-detection\\yolo-object-detection\\images\\soccer.jpg");
        Size sz = new Size(288,288);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);

        while (true) {

            if (cap.read(frame)) {
                Mat blob = Dnn.blobFromImage(frame, 0.00392, sz, new Scalar(0), true, false);
                net.setInput(blob);
                net.forward(result, outBlobNames);
                // outBlobNames.forEach(System.out::println);
                // result.forEach(System.out::println);

                float confThreshold = 0.6f;
                List<Integer> clsIds = new ArrayList<>();
                List<Float> confs = new ArrayList<>();
                List<Rect> rects = new ArrayList<>();
                for (int i = 0; i < result.size(); ++i)
                {
                    Mat level = result.get(i);
                    for (int j = 0; j < level.rows(); ++j)
                    {
                        Mat row = level.row(j);
                        Mat scores = row.colRange(5, level.cols());
                        Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                        float confidence = (float)mm.maxVal;
                        Point classIdPoint = mm.maxLoc;
                        if (confidence > confThreshold)
                        {
                            int centerX = (int)(row.get(0,0)[0] * frame.cols());
                            int centerY = (int)(row.get(0,1)[0] * frame.rows());
                            int width   = (int)(row.get(0,2)[0] * frame.cols());
                            int height  = (int)(row.get(0,3)[0] * frame.rows());
                            int left    = centerX - width  / 2;
                            int top     = centerY - height / 2;

                            clsIds.add((int)classIdPoint.x);
                            confs.add((float)confidence);
                            rects.add(new Rect(left, top, width, height));
                        }
                    }
                }
                float nmsThresh = 0.5f;
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

                int [] ind = indices.toArray();
                int j=0;
                for (int i = 0; i < ind.length; ++i)
                {
                    int idx = ind[i];
                    Rect box = boxesArray[idx];
                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0,0,255), 2);
                    //i=j;

                    System.out.println(idx);
                }
                // Imgcodecs.imwrite("D://out.png", image);
                //System.out.println("Image Loaded");
                ImageIcon image = new ImageIcon(Mat2bufferedImage(frame));
                vidpanel.setIcon(image);
                vidpanel.repaint();
                // System.out.println(j);
                // System.out.println("Done");

            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        System.load("F:\\work-space\\hot_yolov3\\src\\data\\opencv_java411.dll");
        String modelWeights =
                "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3_18000.weights";
        String modelConfiguration =
                "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3.cfg";

        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        //Thread.sleep(5000);

        Mat frame = Imgcodecs.imread("E:\\data\\VOC2007_augment\\JPEGImages\\21.jpg");
        Size sz = new Size(416, 416);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);


        Mat blob = Dnn.blobFromImage(frame, 0.00392, sz, new Scalar(0), true, false);
        net.setInput(blob);
        net.forward(result, outBlobNames);

        float confThreshold = 0.25f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < result.size(); ++i) {
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > confThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);
                    rects.add(new Rect(left, top, width, height));
                }
            }
        }
        float nmsThresh = 0.45f;
        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
        Rect[] boxesArray = rects.toArray(new Rect[0]);
        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
        System.out.println(clsIds);

        List tags = Arrays.asList("diode", "shadow","stain","");
        System.out.println(tags.get(0));

        String ids = "diode,shadow,stain";
        System.out.println(ids.charAt(0));

        int[] ind = indices.toArray();
        int j = 0;
        for (int i = 0; i < ind.length; ++i) {
            int idx = ind[i];
            Rect box = boxesArray[idx];
            Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
            System.out.println(tags.get(idx));
            Imgproc.putText(frame, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);

            //i=j;

            System.out.println(idx);
        }
         Imgcodecs.imwrite("out21.jpg", frame);
        }


    private static BufferedImage Mat2bufferedImage(Mat image) {
        MatOfByte bytemat = new MatOfByte();
        Imgcodecs.imencode(".jpg", image, bytemat);
        byte[] bytes = bytemat.toArray();
        InputStream in = new ByteArrayInputStream(bytes);
        BufferedImage img = null;
        try {
            img = ImageIO.read(in);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return img;
    }
}

