#ifndef __IR_CONTROLLER_HPP__
#define __IR_CONTROLLER_HPP__

#include <algorithm>
#include <unistd.h>
#include <cstring>
#include <string>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "ros/ros.h"
#include "libsonar/libSonar.h"

#define OUT

class IRController {

public:
    IRController();
    virtual ~IRController();
    void init();
    void run(OUT double* dists);
    void finalize();
    void send(double* dists);

private:
    int sock;
    ros::NodeHandle* nh;
    Sonar sonars[4];

    int connectSocket();
    void computePercentage(double* dists, OUT float* perc);
};

#endif