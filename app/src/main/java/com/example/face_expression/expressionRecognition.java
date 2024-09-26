package com.example.face_expression;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import com.example.face_expression.R;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class expressionRecognition {
    private Interpreter interpreter;

    //mengatur ukuran input
    private int input;

    //mengatur ukuran tinggi dan lebar
    private int height=0;
    private int width=0;

    private GpuDelegate gpuDelegate=null;

    private CascadeClassifier cascadeClassifier;

    class Pair {
        int index;
        float probability;

        Pair(int index, float probability) {
            this.index = index;
            this.probability = probability;
        }
    }

    expressionRecognition(AssetManager assetManager, Context context, String modelpath, int inputsize) throws IOException {
        input=inputsize;
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        //options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        //load model weight ke interpreter
        interpreter= new Interpreter(loadModelfile(assetManager,modelpath),options);
        Log.d("Facial Expression","Model sudah dimuat ");

        //load cascade classifier
        try {
            //definisikan file input stream untuk membaca file classifier
            InputStream inputStream=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            //membuat folder
            File cascadedir=context.getDir("cascade",context.MODE_PRIVATE);
            //membuat file cascade classifier
            File mcascadefile=new File(cascadedir,"haarcascade_frontalface");
            //membuat file output stream untuk menulis file cascade classifier
            FileOutputStream outputStream=new FileOutputStream(mcascadefile);
            //membuat buffer untuk membaca data
            byte[] buffer=new byte[4096];
            int byteread;
            //membaca data cascade classifier
            //-1 berarti tidak membaca data apapun
            while ((byteread= inputStream.read(buffer))!=-1){
                //menulis data cascade classifier
                outputStream.write(buffer,0,byteread);
            }
            inputStream.close();
            outputStream.close();
            cascadeClassifier=new CascadeClassifier(mcascadefile.getAbsolutePath());
            Log.d("Facial Expression","Classifier sudah dimuat");

        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public Mat recognizeimage(Mat mat_image){
        //merotasi gambar 90 derajat
        Core.flip(mat_image.t(),mat_image,1);
        long startTime = System.currentTimeMillis();

        //mengubah gambar menjadi gray scale
        Mat grayscale=new Mat();
        Imgproc.cvtColor(mat_image,grayscale,Imgproc.COLOR_RGBA2GRAY);

        height=grayscale.height();
        width=grayscale.width();

        int facesize=(int)(height*0.1);


        MatOfRect face=new MatOfRect();
        if (cascadeClassifier!=null){
            cascadeClassifier.detectMultiScale(grayscale,face,1.1,2,2,new Size(facesize,facesize),new Size());
        }

        Rect[] facearray=face.toArray();
        for (int i=0;i<facearray.length;i++){
            Imgproc.rectangle(mat_image,facearray[i].tl(),facearray[i].br(),new Scalar(0,255,0,255),2);
            Rect roi=new Rect((int)facearray[i].tl().x,(int)facearray[i].tl().y,
                    ((int) facearray[i].br().x)-(int) (facearray[i].tl().x),
                    ((int) facearray[i].br().y)-(int) (facearray[i].tl().y));

            Mat crop_rgba=new Mat(mat_image,roi);

            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(crop_rgba.cols(),crop_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crop_rgba,bitmap);
            Bitmap scaledbitmap=Bitmap.createScaledBitmap(bitmap,48,48,false);
            ByteBuffer byteBuffer=convertbitmap(scaledbitmap);

            float[][] emotion = new float[1][7];
            interpreter.run(byteBuffer, emotion);
            Log.d("Facial Expression", "Output : " + Arrays.toString(emotion[0]));
            String logMessage = "Output: " + Arrays.toString(emotion[0]);
            writeLogToFile(logMessage);

            List<Pair> pairs = new ArrayList<>();
            for (int k = 0; k < emotion[0].length; k++) {
                pairs.add(new Pair(k, emotion[0][k]));
            }

// Sortir berdasarkan probabilitas secara menurun (descending order)
            Collections.sort(pairs, new Comparator<Pair>() {
                @Override
                public int compare(Pair p1, Pair p2) {
                    return Float.compare(p2.probability, p1.probability);
                }
            });
            // Menyimpan 3 ekspresi teratas beserta probabilitasnya
            float[] topProbabilities = new float[3];
            String[] topEmotions = new String[3];

            for (int j = 0; j < 3; j++) {
                topProbabilities[j] = pairs.get(j).probability;
                topEmotions[j] = getemotion(pairs.get(j).index);
            }

// Menampilkan tiga prediksi teratas di ujung atas kiri layar
            Scalar warnafont1 = new Scalar(0, 128, 0);

            int baseLine = 30; // Titik awal di sumbu y untuk baris pertama
            int lineHeight = 40; // Jarak antar baris

            for (int j = 0; j < 3; j++) {  // Loop hanya untuk 3 prediksi teratas
                String text = topEmotions[j] + " (" + String.format("%.2f", topProbabilities[j] * 100) + "%)";
                Imgproc.putText(mat_image, text,
                        new Point(10, baseLine + j * lineHeight),  // Menyesuaikan posisi y untuk setiap baris
                        Core.FONT_HERSHEY_SIMPLEX, 1.0, warnafont1, 2);
            }

            float ekspresi_v = getMaxIndex(emotion[0]);
            Log.d("Facial expression", "Output Index: " + ekspresi_v);
            logMessage = "Output Index: " + ekspresi_v;
            writeLogToFile(logMessage);

            String ekspresi=getemotion(ekspresi_v);
            String log_ekspresi = "Ekpresi: " + ekspresi;
            writeLogToFile(log_ekspresi);

            double ukuran_font = 1.5;
            int Ketebalan = 5;
            Scalar warnafont = new Scalar(255, 255, 255);
            long endTime = System.currentTimeMillis();
            long latency = endTime - startTime;
            Log.d("Facial Expression", "Latency: " + latency + "ms");
            logMessage = "Latency: " + latency + "ms";
            writeLogToFile(logMessage);
            Imgproc.putText(mat_image,  " (" + latency + "ms)",
                    new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, ukuran_font, warnafont, Ketebalan);

        }

        Core.flip(mat_image.t(),mat_image,0);

        return mat_image;
    }
    private int[] getTopIndices(float[] array, int topN) {
        int[] indices = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices);
        return Arrays.copyOfRange(indices, 0, topN);
    }
    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void writeLogToFile(String logMessage) {
        String logFilePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS) + "/expression_logs_shufflenet_takut.txt";
        FileOutputStream logOutputStream = null;

        try {
            logOutputStream = new FileOutputStream(logFilePath, true);
            logOutputStream.write(logMessage.getBytes());
            logOutputStream.write("\n".getBytes());
        } catch (IOException e) {
            Log.e("Facial Expression", "Error writing log to file: " + e.getMessage());
        } finally {
            if (logOutputStream != null) {
                try {
                    logOutputStream.close();
                } catch (IOException e) {
                    // Ignore closing exception
                }
            }
        }
    }

    private String getemotion(float ekspresi_v) {
        String val="";
        if (ekspresi_v>=0 && ekspresi_v<0.5){
            val="Marah";
        }
        else if (ekspresi_v>=0.5 && ekspresi_v<1.5){
            val="Jijik";
        }
        else if (ekspresi_v>=1.5 && ekspresi_v<2.5){
            val="Takut";
        }
        else if (ekspresi_v>=2.5 && ekspresi_v<3.5) {
            val = "Senang";
        }
        else if (ekspresi_v>=3.5 && ekspresi_v<4.5){
            val="netral";
        }
        else if (ekspresi_v>=4.5 && ekspresi_v<5.5) {
            val = "sedih";
        }
        else {
            val="Terkejut";
        }
        return val;
    }

    private ByteBuffer convertbitmap(Bitmap scaledbitmap) {
        ByteBuffer byteBuffer;
        int size_image=input;
        byteBuffer=ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intvalues=new int[size_image*size_image];
        scaledbitmap.getPixels(intvalues,0,scaledbitmap.getWidth(),0,0,scaledbitmap.getWidth(),scaledbitmap.getHeight());
        int pixel=0;
        for (int i=0;i<size_image;++i){
            for (int j=0;j<size_image;++j){
                final int val=intvalues[pixel++];
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);
            }
        }
        return byteBuffer;
    }

    //fungsi untuk load model
    private MappedByteBuffer loadModelfile(AssetManager assetManager, String modelpath) throws IOException{
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelpath);
        //membuat input stream untuk membaca file
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();

        long startoffset=assetFileDescriptor.getStartOffset();
        long declaredlenght=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredlenght);
    }

}
