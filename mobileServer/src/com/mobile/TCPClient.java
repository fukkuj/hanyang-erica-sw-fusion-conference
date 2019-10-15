package com.mobile;

import java.io.*;
import java.net.Socket;

public class Client {

    @SuppressWarnings("deprecation")
    public static void main(String[] args) {

        String hostName = "";
        String trashInfo = "80 80 80 80";

        int port = 8080;

        Socket client = null;
        String rasRequest = " ";
        String data = " ";

        DataInputStream dataInputStream = null;
        PrintStream printStream = null;
        byte[] bs = new byte[2048];
        int i = 0;

        try {
            System.out.println("Connecting to: " + hostName + " on port " + port);
                client = new Socket(hostName, port);
                printStream = new PrintStream(client.getOutputStream());
                printStream.println(trashInfo);
                dataInputStream = new DataInputStream(client.getInputStream());
                System.out.println("Response from the server: ");

                while(( data = dataInputStream.readLine() ) != null) {
                    System.out.println(data);
                }
    } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                dataInputStream.close();
                printStream.close();
                client.close();
            } catch (IOException e) {
                System.out.println("Error Occured: " + e.toString());
            }
        }

    }
}
