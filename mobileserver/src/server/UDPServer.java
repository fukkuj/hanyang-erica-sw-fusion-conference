package server;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 모바일과 통신을 연결하기 위한 UDP 서버
 */
public class UDPServer extends Thread {

    private DatagramSocket server;
    public Queue<UDPRequest> requests;
    public boolean on;

    public UDPServer() throws SocketException {
        server = new DatagramSocket(Environment.UDP_PORT);
        requests = new ConcurrentLinkedQueue<>();
        on = true;
    }

    @Override
    public void run() {
        System.out.println("Start UDP server...");

        // 서버가 종료할 때까지 반복
        while (on) {
            try {
                // 요청 받을 버퍼
                byte[] buf = new byte[16];

                // 요청을 받음
                DatagramPacket packet = new DatagramPacket(buf, buf.length);
                server.receive(packet);

                // 디버깅을 위해
                String data = new String(packet.getData(), 0, packet.getLength()).trim();

                // 디버깅용
                System.out.println("Received: " + data);

                // 요청을 객체화해서 큐에 넣음
                UDPRequest request = new UDPRequest(this, packet.getAddress().getHostAddress(), packet.getPort());
                requests.add(request);

            } catch (IOException exc) {

            }
        }
    }

    /**
     * 현재까지 들어온 요청이 있으면, 요청 1개를 반환한다.
     * @return 요청 객체
     */
    public UDPRequest getOneRequest() {
        if (requests.isEmpty())
            return null;

        return requests.poll();
    }

    /**
     * UDP 클라이언트(모바일 앱)로 답장을 보냄
     * @param host 모바일 앱의 호스트 주소
     * @param port 모바일 앱의 포트번호
     * @param data 데이터 ("1: 34 23 64 89" 처럼 생긴 문자열)
     */
    public void send(String host, int port, String data) {
        try {
            System.out.println("Sending " + data + "...");
            byte[] buf = data.getBytes();
            DatagramPacket packet = new DatagramPacket(buf, buf.length);
            packet.setAddress(InetAddress.getByName(host));
            packet.setPort(port);
            server.send(packet);
        } catch (IOException exc) {
            exc.printStackTrace();
        }
    }

    /**
     * 서버를 중지함
     */
    public void stopServer() {
        on = false;
        server.close();
    }
}
