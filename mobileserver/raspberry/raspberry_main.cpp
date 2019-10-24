#include <iostream>
#include <signal.h>
#include <cstring>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <arpa/inet.h>

void handler(int signo);
int connect();

bool on = true;

int main(int argc, char* argv[])
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    
    sigaction(SIGINT, &sa, NULL);

    int sock = connect();
    if (sock == -1) {
        return -1;
    }

    while (on) {
        char req[4] = {0,};
        const char* buf = "77 88 22 33\n";

        read(sock, req, 3);
        printf("Sending: %s\n", buf);
        write(sock, buf, strlen(buf));
    }

    close(sock);

    return 0;
}

void handler(int signo)
{
    on = false;
}

int connect()
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "CANNOT open a socket.\n";
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(13345);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cerr << "CANNOT connect to server.\n";
        return -1;
    }

    // std::string id = "1\n";
    const char* buf = "1\n";
    write(sock, buf, strlen(buf));

    return sock;
}