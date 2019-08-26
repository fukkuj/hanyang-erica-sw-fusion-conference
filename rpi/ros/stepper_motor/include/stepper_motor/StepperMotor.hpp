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

	const MotorInfo box_motor = {
		BOX_LEFT_MOTOR_CLK, BOX_LEFT_MOTOR_DIR
	};

	const MotorInfo support_motor1 = {
		SUPPORT_MOTOR_CLK, SUPPORT_LEFT_MOTOR_DIR
	};
	const MotorInfo support_motor2 = {
		SUPPORT_MOTOR_CLK, SUPPORT_RIGHT_MOTOR_DIR
	};

	int initial_motor_clock = 2000; // more less,  more faster
	int min_motor_clock = 200;
	int motor_speed_up = 1;
};

#endif
