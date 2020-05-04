package com.yolo3test;

import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.global.opencv_dnn;

import org.bytedeco.opencv.opencv_dnn.*;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;

import java.util.*;
import java.io.*;


public class Yolo3Demo {

//    public void Yolo3(String img_path) {
    public void Yolo3() {

        String img_path = "F:\\maven\\hot_yolov3\\src\\resources\\person.jpg";

        Mat img = imread(img_path);
        String cfg = "F:\\maven\\hot_yolov3\\src\\resources\\yolov4-custom.cfg";
        String model = "F:\\maven\\hot_yolov3\\src\\resources\\yolov4.weights";

        Net net = opencv_dnn.readNetFromDarknet(cfg, model);
        Mat blob = opencv_dnn.blobFromImage(img, 1 / 255.0, new Size(416, 416), new Scalar(0.0), true, false, CV_32F);

        //set preferable
        net.setPreferableBackend(3); //0
            /*
            0:DNN_BACKEND_DEFAULT
            1:DNN_BACKEND_HALIDE
            2:DNN_BACKEND_INFERENCE_ENGINE
            3:DNN_BACKEND_OPENCV
             */
        net.setPreferableTarget(0);
            /*
            0:DNN_TARGET_CPU
            1:DNN_TARGET_OPENCL
            2:DNN_TARGET_OPENCL_FP16
            3:DNN_TARGET_MYRIAD
            4:DNN_TARGET_FPGA
             */

        net.setInput(blob);

        StringVector outNames = net.getUnconnectedOutLayersNames();
        //System.out.println(outNames.toString());

        MatVector outs = new MatVector(outNames.size());

        net.forward(outs, outNames);

        float threshold = 0.25f;
        float nmsThreshold = 0.45f;
        GetResult(outs, img, threshold, nmsThreshold, true,img_path);
        //System.out.println(outs);
    }

    private void GetResult(MatVector output, Mat image, float threshold, float nmsThreshold, boolean nms,String img_path) {
        nms = false;
        IntVector classIds = new IntVector();
        FloatVector confidences = new FloatVector();
        RectVector boxes = new RectVector();
        try {
            for (int i = 0; i < output.size(); ++i) {
                Mat result = output.get(i);
                //System.out.println(result);
                for (int j = 0; j < result.rows(); j++) {
                    FloatPointer data = new FloatPointer(result.row(j).data());
                    Mat scores = result.row(j).colRange(5, result.cols());

                    Point classIdPoint = new Point(1);
                    DoublePointer confidence = new DoublePointer(1);

                    minMaxLoc(scores, null, confidence, null, classIdPoint, null);

                    if (confidence.get() > threshold) {
                        int centerX = (int) (data.get(0) * image.cols());
                        int centerY = (int) (data.get(1) * image.rows());
                        int width = (int) (data.get(2) * image.cols());
                        int height = (int) (data.get(3) * image.rows());
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x());
                        confidences.push_back((float) confidence.get());
                        boxes.push_back(new Rect(left, top, width, height));
                    }
                }
            }

//            if (nms) {
//
//                IntPointer indices = new IntPointer(confidences.size());
//                for (int i = 0; i < confidences.size(); ++i) {
//                    Rect box = boxes.get(i);
//                    int classId = classIds.get(i);
//                    String res = "idx="+classId+"conf="+confidences.get(i);
//                    res += "box.x=" + box.x() +"box.y="+box.y()+"box.width"+box.width()+"box.height"+box.height();
//                    System.out.println(res);
//                }
//            }

            IntPointer indices = new IntPointer(confidences.size());
            FloatPointer confidencesPointer = new FloatPointer(confidences.size());
            confidencesPointer.put(confidences.get());

            NMSBoxes(boxes, confidencesPointer, threshold, nmsThreshold, indices, 1.f, 0);
//            NMSBoxes(boxes, confidencesPointer, threshold, nmsThreshold, indices);

            List<String> list = new ArrayList<String>();
            FileInputStream fis = new FileInputStream("F:\\maven\\hot_yolov3\\src\\resources\\coco.names");
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
            BufferedReader br = new BufferedReader(isr);
            String line;
            while ((line = br.readLine()) != null) {
                list.add(line);
            }
            String[] Labels = list.toArray(new String[list.size()]);
            br.close();
            isr.close();
            fis.close();

            List diode = new ArrayList<>();
            List shadow = new ArrayList<>();
            List stain = new ArrayList<>();
            //System.out.println(indices.sizeof());
            //System.out.println(indices.limit());

            IplImage rawImage = null;
            rawImage = cvLoadImage(img_path);

            for (int m = 0; m < indices.limit(); ++m) {
                int i = indices.get(m);
                Rect box = boxes.get(i);
                //System.out.println(box);
                int classId = classIds.get(i);
                //System.out.println("name:"+Labels);
                //System.out.println("classid"+classId);

                int x1 = box.x();
                int y1 = box.y();
                int x2 = box.x() + box.width();
                int y2 = box.y() + box.height();


                CvPoint pt1 = cvPoint(x1, y1);
                CvPoint pt2 = cvPoint(x2, y2);
                CvScalar color = cvScalar(255, 0, 0, 0);       // blue [green] [red]
                cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                cvPutText(rawImage,Labels[classId],pt1,cvFont(1,1),CvScalar.BLACK);

                /*if (classId == 0) {
                    int x1 = box.x();
                    int y1 = box.y();
                    int x2 = box.x() + box.width();
                    int y2 = box.y() + box.height();


                    CvPoint pt1 = cvPoint(x1, y1);
                    CvPoint pt2 = cvPoint(x2, y2);
                    CvScalar color = cvScalar(255, 0, 0, 0);       // blue [green] [red]
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1,1),CvScalar.BLACK);


                    diode.add(box.x());
                    diode.add(box.y());
                    diode.add(box.width());
                    diode.add(box.height());
                }
                if (classId == 1) {
                    int x1 = box.x();
                    int y1 = box.y();
                    int x2 = box.x() + box.width();
                    int y2 = box.y() + box.height();
                    CvPoint pt1 = cvPoint(x1, y1);
                    CvPoint pt2 = cvPoint(x2, y2);
                    CvScalar color = cvScalar(0, 255, 0, 0);       // blue [green] [red]
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1,1),CvScalar.BLACK);


                    shadow.add(box.x());
                    shadow.add(box.y());
                    shadow.add(box.width());
                    shadow.add(box.height());
                }
                if (classId == 2) {
                    int x1 = box.x();
                    int y1 = box.y();
                    int x2 = box.x() + box.width();
                    int y2 = box.y() + box.height();
                    CvPoint pt1 = cvPoint(x1, y1);
                    CvPoint pt2 = cvPoint(x2, y2);
                    CvScalar color = cvScalar(0, 0, 255, 0);       // blue [green] [red]
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1,1),CvScalar.BLACK);


                    stain.add(box.x());
                    stain.add(box.y());
                    stain.add(box.width());
                    stain.add(box.height());
                }*/
                String res = "idx=" + classId + "  name=" + Labels[classId] + "  conf=" + confidences.get(i);
                res += "  box.x=" + box.x() + "  box.y=" + box.y() + "  box.width=" + box.width() + "  box.height=" + box.height();
                System.out.println(res);
            }
            File tempFile =new File( img_path.trim());
            String fileName = tempFile.getName();
            //System.out.println("fileName:" + fileName);

            String save_img = "result_" + fileName;
            //System.out.println(save_img);
            cvSaveImage(save_img, rawImage);
            //cvSaveImage("result.jpg", rawImage);


            //String total_diode = "diode_num:" + diode.size() / 4 + "  |diode_list:" + diode;
            //String total_shadow = "shadow_num:" + shadow.size() / 4 + "  |shadow_list:" + shadow;
            //String total_stain = "stain_num:" + stain.size() / 4 + "  |stain_list:" + stain;

            //System.out.println(total_diode);
            //System.out.println(total_shadow);
            //System.out.println(total_stain);


        } catch (Exception e) {
            System.out.println("GetResult error:" + e.getMessage());
        }
    }

    public static void main(String[] args) {
//        if (args.length != 1) {
//            System.out.println("Input parameter error!!!\n Please try again!!!");
//            System.exit(0);
//        }
        Yolo3Demo be = new Yolo3Demo();
//        be.Yolo3(args[0]);
        be.Yolo3();

    }
}