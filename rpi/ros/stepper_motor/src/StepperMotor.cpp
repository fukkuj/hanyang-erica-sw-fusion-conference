#include "stepper_motor/StepperMotor.hpp"
#include "stepper_motor/stepper_motor.hpp"
#include "std_srvs/SetBool.h"
#include "wiringPi.h"
#include <thread>
#include <vector>


StepperMotor::StepperMotor(ros::NodeHandle _nh)
	: nh(_nh), is_ready(true)
{

}

StepperMotor::~StepperMotor()
{

}

void StepperMotor::Setup()
{
	this->sub = nh.subscribe("motor", 2,
			&StepperMotor::MotorCallback, this);
	this->serv_clnt = nh.serviceClient<std_srvs::SetBool>("motor_done");
}

void StepperMotor::Destroy()
{
	
}

void StepperMotor::MotorCallback(const std_msgs::Int32MultiArray::ConstPtr& ptr)
{
	if (this->is_ready) {
		this->is_ready = false;

		std::vector<int> data = ptr->data;

		ROS_INFO("Motor: %d", data[0]);
		ROS_INFO("Motor dir: %d", data[1]);
		ROS_INFO("Motor step: %d", data[2]);

		if (data[0] == BOX_MOTOR)
			moveBoxMotor(data[1], data[2]);

		else if (data[0] == SUPPORT_MOTOR)
			moveSupportMotor(data[1], data[2]);

		this->is_ready = true;

		std_srvs::SetBool request;
		request.request.data = true;

		this->serv_clnt.call(request);
	}
}

void StepperMotor::moveBoxMotor(int dir, int step)
{
	digitalWrite(BOX_MOTOR_ENABLE, HIGH);

	if (dir == FORWARD)
		digitalWrite(BOX_MOTOR_DIR, HIGH);
	else
		digitalWrite(BOX_MOTOR_DIR, LOW);

	int motor_clock = this->initial_motor_clock;
	int i = 0;

	for (; i < step/2; i += 1) {
		digitalWrite(BOX_MOTOR_CLK, HIGH);
		delayMicroseconds(motor_clock);
		digitalWrite(BOX_MOTOR_CLK, LOW);
		delayMicroseconds(motor_clock);

		motor_clock -= this->motor_speed_up;
		if (motor_clock < this->min_motor_clock)
			motor_clock = this->min_motor_clock;
	}

	for (; i < step; i += 1) {
		digitalWrite(BOX_MOTOR_CLK, HIGH);
		delayMicroseconds(motor_clock);
		digitalWrite(BOX_MOTOR_CLK, LOW);
		delayMicroseconds(motor_clock);

		motor_clock += this->motor_speed_up;
		if (motor_clock > this->initial_motor_clock)
			motor_clock = this->initial_motor_clock;
	}

	digitalWrite(BOX_MOTOR_ENABLE, LOW);
}

void StepperMotor::moveSupportMotor(int dir, int step)
{
	digitalWrite(SUPPORT_MOTOR_ENABLE, HIGH);
	
	if (dir == FORWARD) {
		digitalWrite(SUPPORT_LEFT_MOTOR_DIR, LOW);
		digitalWrite(SUPPORT_RIGHT_MOTOR_DIR, HIGH);
	}
	else {
		digitalWrite(SUPPORT_LEFT_MOTOR_DIR, HIGH);
		digitalWrite(SUPPORT_RIGHT_MOTOR_DIR, LOW);
	}

	int motor_clock = this->initial_motor_clock;
	int i = 0;

	for (; i < step/2; i += 1) {
		digitalWrite(SUPPORT_MOTOR_CLK, HIGH);
		delayMicroseconds(motor_clock);
		digitalWrite(SUPPORT_MOTOR_CLK, LOW);
		delayMicroseconds(motor_clock);

		motor_clock -= this->motor_speed_up;
		if (motor_clock < this->min_motor_clock)
			motor_clock = this->min_motor_clock;
	}

	for (; i < step; i += 1) {
		digitalWrite(SUPPORT_MOTOR_CLK, HIGH);
		delayMicroseconds(motor_clock);
		digitalWrite(SUPPORT_MOTOR_CLK, LOW);
		delayMicroseconds(motor_clock);

		motor_clock += this->motor_speed_up;
		if (motor_clock > this->initial_motor_clock)
			motor_clock = this->initial_motor_clock;
	}

	digitalWrite(SUPPORT_MOTOR_ENABLE, LOW);
}
