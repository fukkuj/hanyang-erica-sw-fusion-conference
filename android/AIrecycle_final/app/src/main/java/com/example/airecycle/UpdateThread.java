package com.example.airecycle;

import android.view.View;

import com.github.mikephil.charting.data.BarEntry;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.LinkedList;
import java.util.List;

public class UpdateThread implements Runnable {

    public static final int INTERVAL = 1000;
    public static final String HOST = "35.229.136.239";
    public static final int PORT = 13346;

    private boolean on;

    private View view;
    private List<BarEntry> entries;
    private DatagramSocket client;
    private String[] splited;

    public UpdateThread(View view, List<BarEntry> entries) throws IOException {
        this.view = view;
        this.entries = entries;

        this.client = new DatagramSocket(14444);
        this.on = true;
    }

    @Override
    public void run() {
        long update_time = System.currentTimeMillis();

        try {
            while (on) {
                if (System.currentTimeMillis() - update_time > INTERVAL) {
                    request();
                    List<String> data = new LinkedList<>();
                    do {
                        data.add(respond());
                    } while (!data.get(data.size() - 1).equals("end"));
                    data.remove(data.size() - 1);

                    for (int i = 0; i < data.size(); i += 1) {
                        System.out.println("Data: " + data.get(i));
                        updateBar(data.get(i));
                    }

                    update_time = System.currentTimeMillis();
                } else {
                    Thread.yield();
                }
            }
        } catch (IOException exc) {
            exc.printStackTrace();
        }
    }

    private void request() throws IOException {
        String req = "request";

        DatagramPacket packet = new DatagramPacket(req.getBytes(), req.getBytes().length);
        packet.setAddress(InetAddress.getByName(HOST));
        packet.setPort(PORT);

        client.send(packet);
    }

    private String respond() throws IOException {
        byte[] response = new byte[16];
        DatagramPacket packet = new DatagramPacket(response, response.length);
        client.receive(packet);
        return new String(packet.getData(), 0, packet.getLength()).trim();
    }

    private void updateBar(String data) {
        splited = data.split(" ");

        for (int i = 1; i < 5; i += 1) {
            entries.get(i-1).setY(Integer.parseInt(splited[i]));
        }


        this.view.postInvalidate();
    }

    public int getCount(){
        int count = 0;
        for (BarEntry entry : entries) {
            if (entry.getY() >= 75) {
                count++;
            }
        }
        return count;
    }
}
