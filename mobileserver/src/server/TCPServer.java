package server;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.LinkedList;
import java.util.List;

/**
 * 쓰레기통과 통신을 맺기 위한 TCP 서버 쓰레드
 */
public class TCPServer extends Thread {

    private ServerSocket server;
    private List<TCPClient> clients;
    private boolean on;

    public TCPServer() throws IOException {
        server = new ServerSocket(Environment.TCP_PORT);
        clients = new LinkedList<>();
        on = true;
    }

    @Override
    public void run() {
        System.out.println("Start TCP server...");

        // 서버가 종료될 때까지 반복
        try {
            while (on) {
                // 쓰레기통(라즈베리)으로부터 통신 연결 수락
                Socket socket = server.accept();

                // 쓰레기통(라즈베리)을 위한 객체 생성
                TCPClient client = new TCPClient(this, socket);

                // 쓰레기통이 여러개일 경우, 관리의 편의를 위해 모두 리스트로 저장
                clients.add(client);

                // 연결 완료된 쓰레기통의 통신 시작.
                client.start();
            }
        } catch (IOException exc) {
            exc.printStackTrace();
        }

    }

    /**
     * 서버 중지
     * @throws IOException
     */
    public void stopServer() throws IOException {
        on = false;
        for (TCPClient client : clients)
            client.stopClient();
        server.close();
    }

    /**
     * 클라이언트 제거
     * @param client
     */
    public void removeClient(TCPClient client) {
        clients.remove(client);
    }

    /**
     * 쓰레기통들의 리스트를 돌면서 찬 정도를 반환한다.
     * @return 각 쓰레기통 정보의 리스트
     */
    public List<String> getClients() {
        List<String> data = new LinkedList<>();
        for (TCPClient client : clients)
            data.add(client.getClientId() + ": " + client.getData());

        return data;
    }
}
