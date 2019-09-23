package com.mobile;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.Arrays;

public class UDPServer {
    public UDPServer(int port) {
        try {
            DatagramSocket ds = new DatagramSocket(port);
            while (true) {
                byte buffer[] = new byte[128];
                Arrays.fill(buffer, (byte) 0);
                DatagramPacket dp = new DatagramPacket(buffer, buffer.length);
                System.out.println("ready");
                ds.receive(dp);
                String str = new String(dp.getData()).trim();
                System.out.println("수신된 데이터 : " + str);

                InetAddress ia = dp.getAddress();
                port = dp.getPort();
                System.out.println("client ip : " + ia + " , client port : " + port);
                dp = new DatagramPacket(dp.getData(),dp.getData().length, ia,port);
                ds.send(dp);
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        new UDPServer(3000);

    }


}
