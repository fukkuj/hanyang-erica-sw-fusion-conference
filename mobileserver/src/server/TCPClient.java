package server;

import java.io.*;
import java.net.Socket;

/**
 * 라즈베리를 위한 클라이언트 클래스.
 * 라즈베리와 통신하게 된다.
 */
public class TCPClient extends Thread {

    private TCPServer server;
    private Socket socket;
    private String data;
    private boolean on;
    private int id;

    private boolean lock;

    public TCPClient(TCPServer server, Socket socket) {
        this.server = server;
        this.socket = socket;
        this.on = true;

        this.lock = false;
    }

    @Override
    public void run() {
        BufferedReader in;
        BufferedWriter out;

        try {
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));

            id = Integer.parseInt(in.readLine());

            System.out.println("Connected Trash Can ID: " + id);

            long previousTime = System.currentTimeMillis() - 1000*Environment.TIME_INTERVAL;

            while (on) {

                // 현재시간 받아옴.
                long currentTime = System.currentTimeMillis();

                // 지정된 시간 간격이 지나지 않았으면, CPU를 다른 쓰레드에게 넘기고 위로 되돌아감.
                if (currentTime - previousTime < Environment.TIME_INTERVAL) {
                    Thread.yield();
                    continue;
                }

                // 데이터를 갱신할때, Lock을 걸어준다.
                this.lock = true;
                out.write("GET", 0, 3);
                out.flush();
                this.data = in.readLine();
                this.lock = false;

                // 시간 업데이트
                previousTime = currentTime;
            }

        } catch (IOException exc) {
            System.out.println("CANNOT connect to trash can ID: " + id);
            try {
                socket.close();
            } catch (IOException e) {

            }
            server.removeClient(this);
        }
    }

    /**
     * 쓰레기통과 연결을 끊는다.
     * @throws IOException
     */
    public void stopClient() throws IOException {
        on = false;
        socket.close();
    }

    /**
     * 쓰레기통으로부터 얻어온 데이터를 저장해뒀다가
     * 이 메소드 호출시 반환
     * @return
     */
    public String getData() {
        // Lock 걸린 동안 기다린다.
        while (lock) { Thread.yield(); }

        return data;
    }

    /**
     * 쓰레기통의 ID값 반환
     * @return
     */
    public int getClientId() {
        return id;
    }
}
