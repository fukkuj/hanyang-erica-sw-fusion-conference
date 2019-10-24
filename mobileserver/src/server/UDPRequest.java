package server;

/**
 * 모바일 앱으로부터 온 요청을 객체화하기 위한 클래스
 */
public class UDPRequest {

    private UDPServer server;
    private String host;
    private int port;

    public UDPRequest(UDPServer server, String host, int port) {
        this.server = server;
        this.host = host;
        this.port = port;
    }

    /**
     * 답장을 보낼때 이놈을 호출.
     * @param data 데이터 문자열 ("1: 32 43 26 89" 처럼 생김)
     */
    public void respond(String data) {
        this.server.send(host, port, data);
    }
}
