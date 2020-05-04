//package com.yolo3test;
//
//import java.net.URL;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//
////import org.opencv.core.*;
////import org.opencv.dnn.Dnn;
////import org.opencv.dnn.Net;
////import org.opencv.utils.Converters;
////import org.opencv.imgcodecs.Imgcodecs;
////import org.opencv.imgproc.Imgproc;
////
////import static org.opencv.imgcodecs.Imgcodecs.imread;
////import org.bytedeco.opencv.global.opencv_imgcodecs;
//
//
//import org.bytedeco.javacpp.*;
//import org.bytedeco.opencv.global.opencv_dnn;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.javacpp.indexer.*;
//import org.bytedeco.opencv.opencv_dnn.Net;
//
//import static org.bytedeco.opencv.global.opencv_core.*;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
//
//
//public class TEST {
//
//    private static List<String> getOutputNames(Net net) {
//        List<String> names = new ArrayList<>();
//
//        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
//
//        System.out.println(outLayers);
//        System.exit(0);
//        List<String> layersNames = net.getLayerNames();
//
//        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
//        return names;
//    }
//
//    public static void main(String[] args) throws InterruptedException {
//        System.load("F:\\maven\\hot_yolov3\\target\\classes\\opencv_java.dll");
//        String modelWeights = "F:\\work-space\\hot_yolov3\\src\\resources\\hot-yolov3.weights";
//        String modelConfiguration = "F:\\work-space\\hot_yolov3\\src\\resources\\hot-yolov3.cfg";
//        Mat src = imread("F:\\maven\\hot_yolov3\\src\\resources\\20190520_132858.jpg");
//
//        URL resource2 = Test01.class.getClassLoader().getResource("");
//        System.out.println(resource2);
//        System.exit(0);
////        URL resource2 = TestASM.class.getClassLoader().getResource("");
//
//
//        Net net = opencv_dnn.readNetFromDarknet(modelConfiguration, modelWeights);
//
//        Size sz = new Size(416, 416);
//
//        MatVector result = new MatVector();
////        List<String> outBlobNames = getOutputNames(net);
//
//        Mat blob = opencv_dnn.blobFromImage(src, 0.00392, sz, new Scalar(0), true, false,CV_32F);
//        net.setInput(blob);
//        StringVector outname = net.getUnconnectedOutLayersNames();
//        System.out.println(outname);
//        net.forward(result, outname);
//        System.out.println(result);
//
//        float confThreshold = 0.25f; // 0.5f;
//        List<Integer> clsIds = new ArrayList<>();
//        List<Float> confs = new ArrayList<>();
//        List<Rect> rects = new ArrayList<>();
//        for (int i = 0; i < result.size(); ++i) {
//            // [cx, cy, w, h]
//            Mat level = result.get(i);
//            System.out.println(level);
//            for (int j = 0; j < level.rows(); ++j) {
//                Mat row = level.row(j);
//                System.out.println(row);
//                Mat scores = row.colRange(5, level.cols());
//                System.out.println(scores);
//                System.exit(0);
//                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
////                System.out.println(mm.toString());
//                float confidence = (float) mm.maxVal;
//                Point classIdPoint = mm.maxLoc;
////                System.out.println(confidence);
//                if (confidence > confThreshold) {
//                    int centerX = (int) (row.get(0, 0)[0] * src.cols());
//                    int centerY = (int) (row.get(0, 1)[0] * src.rows());
//                    int width = (int) (row.get(0, 2)[0] * src.cols());
//                    int height = (int) (row.get(0, 3)[0] * src.rows());
//                    int left = centerX - width / 2;
//                    int top = centerY - height / 2;
//
//                    clsIds.add((int) classIdPoint.x);
//                    confs.add((float) confidence);
//                    rects.add(new Rect(left, top, width, height));
//                }
//            }
//        }
//        float nmsThresh = 0.45f;  // 0.5f
//        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
//        Rect[] boxesArray = rects.toArray(new Rect[0]);
//        MatOfRect boxes = new MatOfRect(boxesArray);
//        MatOfInt indices = new MatOfInt();
//        System.out.println(boxes.toString());
//        System.out.println(boxes.size());
//        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
//
//        List tags = Arrays.asList("diode", "shadow","stain","");
//
//        int[] ind = indices.toArray();
//        int j = 0;
//        for (int i = 0; i < ind.length; ++i) {
//            int idx = ind[i];
//            Rect box = boxesArray[idx];
//            if (clsIds.get(i) == 0) {
//                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
//                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
////                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
//            } else if (clsIds.get(i) == 27) {
//                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
//                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
//            } else {
//                Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
//                Imgproc.putText(src, tags.get(clsIds.get(idx)).toString(), box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255,0,0),2);
//            }
//
////            System.out.println(idx);
////            System.out.println(clsIds);
////            System.out.println(confs);
////            System.out.println(box);
////            System.out.println(box.tl());
////            System.out.println(box.br());
//            System.out.println(tags.get(clsIds.get(idx)) + " | " + clsIds.get(idx));
//
//        }
//        Imgcodecs.imwrite("out-20190520_132858.jpg", src);
//
//
//    }
//
//}
