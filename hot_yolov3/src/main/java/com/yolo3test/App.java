package com.yolo3test;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static org.opencv.imgcodecs.Imgcodecs.imread;


public class App {

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("NATIVE_LIBRARY_NAME:" + Core.NATIVE_LIBRARY_NAME);
        System.load("F:\\work-space\\hot_yolov3\\src\\data\\opencv_java411.dll");
        String modelWeights = "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3_18000.weights";
        String modelConfiguration = "F:\\work-space\\hot_yolov3\\src\\data\\hot-yolov3.cfg";

        String tfyolo = "F:\\work-space\\hot_yolov3\\src\\data\\yolov3_base30_211.pb";
//        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
//        Net net = Dnn.readNetFromTensorflow(tfyolo);
        Net net = Dnn.readNet(tfyolo);
        // Thread.sleep(5000);

        Size sz = new Size(416, 416);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);

        //     while (true) {
        //        if (cap.read(frame)) {
        Mat src = imread("E:\\data\\VOC2007_augment\\JPEGImages\\21.jpg");

        Mat blob = Dnn.blobFromImage(src, 0.00392, sz, new Scalar(0), true, false);
        net.setInput(blob);
        net.forward(result, outBlobNames);

        // outBlobNames.forEach(System.out::println);
        // result.forEach(System.out::println);
        float confThreshold = 0.25f; // 0.5f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < result.size(); ++i) {
            // [center_x, center_y, width, height]
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * src.cols());
                    int centerY = (int) (row.get(0, 1)[0] * src.rows());
                    int width = (int) (row.get(0, 2)[0] * src.cols());
                    int height = (int) (row.get(0, 3)[0] * src.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);
                    rects.add(new Rect(left, top, width, height));
                }
            }
        }
        float nmsThresh = 0.45f;  // 0.5f
        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
        Rect[] boxesArray = rects.toArray(new Rect[0]);
        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

        List tags = Arrays.asList("diode", "shadow","stain","");

        int[] ind = indices.toArray();
        int j = 0;
        for (int i = 0; i < ind.length; ++i) {
            int idx = ind[i];
            Rect box = boxesArray[idx];
            if (clsIds.get(i) == 0) {
                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
//                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            } else if (clsIds.get(i) == 27) {
                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
            } else {
                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
            }
            //i=j;
//            System.out.println(idx);
//            System.out.println(clsIds);
//            System.out.println(confs);
//            System.out.println(box);
//            System.out.println(box.tl());
//            System.out.println(box.br());
            System.out.println(tags.get(clsIds.get(idx)) + " | " + confs.get(i) + "%" + " | " + clsIds.get(idx));

        }
        Imgcodecs.imwrite("out21.png", src);


    }

}
