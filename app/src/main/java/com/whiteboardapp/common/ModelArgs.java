package com.whiteboardapp.common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ModelArgs {
    public float conf;
    public float iou;
    public boolean agnosticNMS;
    public int maxDet;
    public int nc;
    //ret sikker på den ikke er nødvendig
    public String classes;

    private final String FILE_NAME = "args.txt";


    //Skal omskrives til at læse en fil
    public ModelArgs() {
        conf = 0.25f;
        iou = 0.7f;
        agnosticNMS = false;
        maxDet = 300;
        nc = 1;
        classes = null;
        LoadModelArgsFromFile();
    }

    public void LoadModelArgsFromFile(){
        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(FILE_NAME));
            String line = reader.readLine();

            while(line != null){
                AssignValues(line);
                line = reader.readLine();
            }
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    private void AssignValues(String line){
        String[] splitLine = line.split(":");
        switch(splitLine[0]){
            case "conf":
                    conf = Float.parseFloat(splitLine[1]);
                break;
            case "iou":
                iou = Float.parseFloat(splitLine[1]);
                break;
            case "agnost_nms":
                agnosticNMS = Boolean.valueOf(splitLine[1]);
                break;
            case "max_det":
                maxDet = Integer.parseInt(splitLine[1]);
                break;
            case "nc":
                nc = Integer.parseInt(splitLine[1]);
                break;
            case "classes":
                classes = null;
                break;
            default:
                System.out.println("How did we even get here? ModelArgs broke");

        }
    }
}
