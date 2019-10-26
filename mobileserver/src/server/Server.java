package server;

import java.io.IOException;
import java.net.SocketException;
import java.util.Scanner;

public class Server extends Thread {

    private TCPServer tcpServer;
    private UDPServer udpServer;

    private boolean on;

    public Server() throws SocketException, IOException {
        tcpServer = new TCPServer();
        udpServer = new UDPServer();
        on = true;
    }

    @Override
    public void run() {
        // TCP, UDP 서버를 각각 시작
        tcpServer.start();
        udpServer.start();

        // 서버가 종료될 때 까지 반복
        while (on) {
            // 모바일로부터 UDP 요청이 들어왔는지 확인
            UDPRequest request = udpServer.getOneRequest();

            // 요청이 없는 경우, CPU를 양보하고 루프 첫 위치로 돌아감
            if (request == null) {
                Thread.yield();
                continue;
            }

            // 요청이 있는 경우, tcp socket으로 연결된 쓰레기통들(라즈베리파이들)의 정보를 얻어와서 UDP로 전송
            for (String data : tcpServer.getClients()) {
                // 데이터 전송
                request.respond(data);
            }

            request.respond("end");
        }
    }

    public void stopServer() {
        try {
            on = false;
            tcpServer.stopServer();
            udpServer.stopServer();
        } catch (IOException exc) {
            exc.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            // 서버 생성 (쓰레드)
            Server server = new Server();
            server.start();

            // 아무거나 입력하면 서버 종료됨.
            System.out.println("Enter e to exit...");
            Scanner sc = new Scanner(System.in);
            sc.nextLine();

            server.stopServer();
        } catch (Exception exc) {
            exc.printStackTrace();
        }
    }
}
