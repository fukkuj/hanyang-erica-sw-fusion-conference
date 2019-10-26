#ifndef __STEPPER_MOTOR_HPP__
#define __STEPPER_MOTOR_HPP__

#include "stepper_motor/MotorInfo.hpp"
#include "stepper_motor/stepper_motor.hpp"
#include "ros/ros.h"
#include "std_msgs/Int32MultiArray.h"

#define FORWARD 1
#define BACKWARD 0

class StepperMotor {
public:
	StepperMotor(ros::NodeHandle);
	virtual ~StepperMotor();

	void Setup();
	void Destroy();
	void MotorCallback(const std_msgs::Int32MultiArray::ConstPtr& ptr);

	void moveBoxMotor(int dir, int step);
	void moveSupportMotor(int dir, int step);

private:
	ros::NodeHandle nh;
	ros::Subscriber sub;
	ros::ServiceClient serv_clnt;

	bool is_ready;

<<<<<<< HEAD
	int box_motor_clock = 1500;
	int support_motor_clock = 500;
=======
	int initial_motor_clock = 1000; // more less,  more faster
	int min_motor_clock = 200;
>>>>>>> 58571d3647cc66d15aef7a657f1292ee4f22a7e2
	int motor_speed_up = 1;
};

#endif
