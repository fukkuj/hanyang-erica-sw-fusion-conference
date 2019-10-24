package client;

import java.io.IOException;
import java.net.*;
import java.util.Scanner;

public class AndroidTestClient {

    private static final String host = "localhost";
    private static final int port = 13346;

    private DatagramSocket socket;

    public AndroidTestClient() throws SocketException {
        socket = new DatagramSocket(14444);
    }

    public void request() throws IOException {
        String req = "request";

        DatagramPacket packet = new DatagramPacket(req.getBytes(), req.getBytes().length);
        packet.setAddress(InetAddress.getByName(host));
        packet.setPort(port);

        socket.send(packet);
    }

    public String respond() throws IOException {
        byte[] response = new byte[16];
        DatagramPacket packet = new DatagramPacket(response, response.length);
        socket.receive(packet);
        return new String(packet.getData(), 0, packet.getLength()).trim();
    }

    public static void main(String[] args) {

        try {
            AndroidTestClient android = new AndroidTestClient();

            while (true) {
                System.out.print("Enter to continue ('e' to exit): ");
                Scanner sc = new Scanner(System.in);
                String input = sc.nextLine();

                if (input.toLowerCase().equals("e"))
                    break;

                android.request();
                String data = android.respond();

                System.out.println("Data: " + data);
            }
        } catch (Exception exc) {
            exc.printStackTrace();
        }
    }
}
