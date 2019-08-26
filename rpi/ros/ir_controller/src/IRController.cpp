#include "ir_controller/IRController.hpp"
#include "ir_controller/ir_controller_node.hpp"
#include <sys/socket.h>
#include <sys/types.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>


IRController::IRController()
    : sock(0), nh(nullptr)
{

}

IRController::~IRController()
{

}

void IRController::init()
{
    ROS_INFO("IR Controller init...");

    this->nh = new ros::NodeHandle();
    for (int i = 0; i < 4; i += 1) {
        this->sonars[i].init(TRIGGER[i], ECHO[i]);
    }

    if (this->connectSocket() < 0) {

    }
}

void IRController::finalize()
{
    delete this->nh;
}

// unit: cm
void IRController::run(OUT double* dists)
{
    for (int i = 0; i < 4; i += 1) {
        dists[i] = sonars[i].distance(TIME_OUT);
    }
}

void IRController::send(double* dists)
{
    float percentages[4];
    this->computePercentage(dists, OUT percentages);

    std::stringstream msg_buffer;
    std::for_each(std::begin(percentages), std::end(percentages), [&msg_buffer](float elem) {
        msg_buffer << elem << " ";
    });

    std::string msg;
    msg_buffer >> msg;
    const char* str = msg.c_str();

    ROS_INFO("Send Message: %s", str);

    write(this->sock, str, strlen(str));
}

void IRController::computePercentage(double* dists, OUT float* perc)
{
    for (int i = 0; i < 4; i += 1) {
        if (dists[i] > 40.0)
            perc[i] = 0.f;
        else if (dists[i] < 10.0)
            perc[i] = 1.f;
        else
            perc[i] = (40.f - dists[i]) / 30.f;
    }
}

int IRController::connectSocket()
{
    struct sockaddr_in addr;

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ADDR);
    addr.sin_port = htons(PORT);

    this->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (this->sock < 0) {
        ROS_ERROR("CANNOT open a socket.");
        return -1;
    }

    if (connect(this->sock, (struct sockaddr*) &addr, sizeof(addr)) < 0) {
        ROS_ERROR("CANNOT connect to server.");
        return -1;
    }

    ROS_INFO("Socket opened successfully.");

    return 0;
}
