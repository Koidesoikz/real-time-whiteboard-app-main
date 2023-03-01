package com.whiteboardapp.common;

import static com.whiteboardapp.common.DebugTags.ImageConversionTag;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class CustomLogger {
    public String output = "";
    private List<Long> SubTimes = new ArrayList<>();
    private List<String> SubTimeNames = new ArrayList<>();

    public void Log(String tag) {
        long totalTime = 0;

        for(int i = 0; i < SubTimes.size(); i ++) {
            totalTime += SubTimes.get(i);
        }

        output = "TotalTime-" + totalTime;

        for(int i = 0; i < SubTimes.size(); i++) {
            output += ":" + SubTimeNames.get(i) + "-" + SubTimes.get(i);
        }

        Log.d(tag, output);
    }

    public void AddTime(long time, String name) {
        SubTimes.add(time);
        SubTimeNames.add(name);
    }
}
