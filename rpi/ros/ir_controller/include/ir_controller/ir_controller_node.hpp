#ifndef __IR_CONTROLLER_NODE_HPP__
#define __IR_CONTROLLER_NODE_HPP__

#include <unistd.h>
#include <signal.h>
#include <cstring>
#include "ros/ros.h"
#include "wiringPi.h"
#include "libsonar/libSonar.h"

const int TRIGGER[] = {1, 2, 3, 4};
const int ECHO[] = {5, 6, 7, 8};

#define ADDR "35.229.136.239"
#define PORT 13345

#define TIME_OUT 30000

#endif