package com.mobile;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;

class Server extends Thread {

    Socket client;
    PrintStream printStream;
    DataInputStream dataInputStream;

    Server (Socket client) {
        this.client = client;
    }

    @SuppressWarnings("deprecation")
    public void run() {
        String response = "";
        String request = "";
        String temp[];

        byte[] b = new byte[2048];
        int i;

        try {
            System.out.println("New connection created: " + client.getRemoteSocketAddress() );
            dataInputStream = new DataInputStream(client.getInputStream());
            request = dataInputStream.readLine();
            printStream = new PrintStream(client.getOutputStream());
            printStream.println(request);
            System.out.println("request :" + request);
            Thread.sleep(1000);
            System.out.println("Connection: " + client.getRemoteSocketAddress() + "closed");


            } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                printStream.close();
                dataInputStream.close();
                client.close();
            } catch ( Exception e) {
                e.printStackTrace();
            }
        }
        RasberryServer.remove(this);
    }
}
public class RasberryServer{

    private static ArrayList<Server> RasList;
    static {
        RasList = new ArrayList<Server>();
    }
    public static void remove(Server RS) {
        RasList.remove(RS);
    }

    private static void stopResources (ServerSocket serverSocket) {
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println("closing the all thread");
                    for (int i = 0; i < RasList.size() ; i++) {
                        Server server = (Server) RasList.get(i);
                        server.printStream.close();
                        server.dataInputStream.close();
                        server.client.close();
                        System.out.println("shutting down the Server");
                        Thread.sleep(1500);
                        serverSocket.close();
                    }

                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }));
    }

    public static void main(String[] args) {
        int port = 0;
        if(args.length != 1) {
            System.out.println("Please enter in the correct format ");
            return ;
        } else {
            port = Integer.parseInt(args[0]);
        }

        Server connection;
        ServerSocket serverSocket = null;
        Socket client = null;
        try {
            serverSocket = new ServerSocket(port);
            stopResources(serverSocket);
            System.out.println("Server is Ready");
            while (true) {
                client = serverSocket.accept();
                if(client != null) {
                    connection = new Server(client);
                    RasList.add(connection);
                    connection.start();
                } else {
                    System.out.println("Something Wrong System exit");
                    return;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
