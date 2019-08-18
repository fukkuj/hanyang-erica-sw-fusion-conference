
#include "ir_controller/ir_controller_node.hpp"
#include "ir_controller/IRController.hpp"

void setupWiringPi();
void interrupt();

IRController* ir_ctrl;

int main(int argc, char* argv[])
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigaddset(&sa_mask, SIGINT);
    sa.sa_handler = interrupt;

    sigaction(SIGINT, &sa, nullptr);

    setupWiringPi();
    ros::init(argc, argv, "ir_controller_node");

    ir_ctrl = new IRController();
    ir_ctrl.init();

    ros::Rate rate(10);

    double dists[4];

    while (ros::ok()) {

        ir_ctrl.run(dists);

        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}

void setupWiringPi()
{
    if (wiringPiSetup() < 0) {
        ROS_ERROR("Wiring pi error!");
        kill(getpid(), SIGINT);
    }
}

void interrupt(int signo)
{
    ROS_INFO("Receive Interrupt...");
    ir_ctrl.finalize();
    ros::shutdown();
}
