package com.mobile;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Date;


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

//        String method = "";
//        String lines = "";
//        String filename = "";
//        String path = "Server Directory";


        byte[] b = new byte[2048];
        int i;

        try {
            System.out.println("New connection created: " + client.getRemoteSocketAddress() );
            dataInputStream = new DataInputStream(client.getInputStream());
            request = dataInputStream.readLine();
//            dataInputStream.readLine();
//            dataInputStream.readLine();

//            lines = request.split("\n")[0];
//            temp = lines.split(" ");
//            method = temp[0];
//            temp = temp[1].split("/");
//            filename = temp[temp.length - 1];
//            File f = new File(path);
//            if(!f.exists()) {
//                f.mkdir();
//            }
//
//            path = path + "/" + filename;
//            f = new File(path);

////            if("get".equalsIgnoreCase(method) ) {
//                if (f.exists()) {
//                    response = "HTTP/1.1 200 OK \r\nDate: " + new Date() + "\r\n";
//                    FileInputStream fileInputStream = new FileInputStream(path);
//                    while ((i = fileInputStream.read(b)) != -1) {
//                        response += new String(b, 0, i) + "\n";
//                    }
//                    fileInputStream.close();
//                } else {
//                    response = "HTTP/1.1 404 Not Found\r\nDate " + new Date() + "\r\n";
//                }
////            }
//        else if ("put".equalsIgnoreCase(method)) {
//                    f = new File(path);
//                    if (!f.exists()) {
//                        f.createNewFile();
//                    }
//                    int size = Integer.parseInt(dataInputStream.readLine());
//                    FileOutputStream fileOutputStream = new FileOutputStream(path);
//                    while (size > 0) {
//                        i = dataInputStream.read(b);
//                        fileOutputStream.write(b, 0, i);
//                        size--;
//                    }
//                    fileOutputStream.close();
//                    response = "HTTP/1.1 200 OK File Created Succesfully\r\nDate: " + new Date() + "\r\n\r\n";
//                } else {
//                    response = "HTTP/1.1 301 Bad Request\r\nDate: " + new Date() + "\r\n\r\n";
//                }


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
