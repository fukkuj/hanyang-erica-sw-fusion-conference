#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include "ros/ros.h"
#include "std_msgs/UInt8MultiArray.h"
#include "usb_camera/UsbCamera.hpp"
#include "usb_camera/usb_camera_node.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef cv::Point3_<uint8_t> Pixel;

/**
 * Constructor
 * 
 * Arguments:
 * @handle NodeHandle object in ros platform
 */
UsbCamera::UsbCamera(ros::NodeHandle& handle)
	: nh(handle), stop(false), cap, isReady(true)
{

}

/**
 * Desctructor
 */
UsbCamera::~UsbCamera()
{
	// free opencv camera object
	delete cap;
}

/**
 * initialization method of this object.
 */
void UsbCamera::init()
{
	// create publisher to send message to other ros node.
	pub = nh.advertise<std_msgs::UInt8MultiArray>(
		"image_data",
		0
	);

	// create service server to receive ready message
	cam_done_serv_server = nh.advertiseService("camera_ready", &UsbCamera::readyService, this);

	// open 2 cameras.
	cap = new cv::VideoCapture(0);

	// input initial image from 2 cameras.
	cv::Mat temp;
	*cap >> temp;

	// initialize initial_frame. this is used to detect whether there is trash in the box.
	cv::resize(temp, initial_frame, cv::Size(HEIGHT, WIDTH));

	ROS_INFO("UsbCamera node is starting...");
}

/**
 * Before destroy UsbCamera object, call this function to free up this object.
 * If you need to do something before destroy this object, you should put them into this method.
 * 
 * UsbCamera 객체를 파괴하기 전에 이 메소드를 호출한다.
 * 만약, 이 객체가 파괴되기 전에 해야 할 작업이 있다면 여기에 추가바람.
 */
void UsbCamera::finalize()
{
	// ?? 이걸 왜 넣었더라
	stop = true;
}

/*
void UsbCamera::resume()
{
	ROS_INFO("Usb camera node resumed.");
	wait = false;
}

bool UsbCamera::isWaiting()
{
	return wait;
}
*/

/**
 * If all is ready to process next trash, other node should send message to this node (usb_camera_node)
 * This message is form of service.
 * After receiving service, call this method to resume camera capture.
 * 
 * 만약, 이전 쓰레기 처리가 끝나서 다음 쓰레기를 처리할 준비가 됬다면, 다른 노드가 이 노드(usb_camera_node)에게
 * 서비스 형태의 메시지를 보낼 것이다. 서비스를 받고, 이 메소드를 호출하게 되는데, 다음 쓰레기를 처리하기 위해 카메라를 resume한다.
 */
bool UsbCamera::readyService(std_srvs::SetBool::Request& req,
			     			 std_srvs::SetBool::Response& res)
{
	ROS_INFO("Camera Done");

	// 
	if (req.data) {
		res.success = true;
		this->isReady = true;
	}

	return true;
}

/**
 * compute main operation of this object.
 * camera capture, compute valuability(is it really trash, not empty box?), and publish.
 * 
 * 이 객체의 메인 기능을 수행하는 메소드.
 * 사진을 찍고, 그 사진이 가치있는가(빈 박스 찍은건 아닌가), 그리고 메시지를 송신.
 */
void UsbCamera::compute()
{
	static int time_count = 0;
	static int send_count = 0;

	// if some error in opening cameras, terminates.
	if (!cap->isOpened()) {
		ROS_ERROR("CANNOT open camera");
		finalize();
		return;
	}

	// if it is not ready to process next trash, return.
	if (!this->isReady)
		return;

	// image matrices
	cv::Mat frame;
	cv::Mat bgrFrame;
	cv::Mat temp;

	// capture images.
	*cap >> frame;

	// resize image into (HEIGHT, WIDTH)
	cv::resize(frame, bgrFrame, cv::Size(HEIGHT, WIDTH));

	// compute the valuabilities of frames.
	Valuable valuable = isValuableFrame(bgrFrame);

	// if valuable, send images to other node
	if (isReady && (send_count > 0 || valuable == Valuable::HIGH)) {
		publish(bgrFrame);
		send_count += 1;
		if (send_count == 8) {
			send_count = 0;
			isReady = false;
		}
	}
	// update initial frame
	else if (valuable == Valuable::LOW) {
		updateInitialFrame(bgrFrame);
	}
}

/**
 * publish(send) images to other node
 * 이미지를 다른 노드에게 송신하기 위한 메소드
 * 
 * Arguments:
 * @frame1 frame from first camera
 * @frame2 frame from second camera
 */
void UsbCamera::publish(cv::Mat& frame)
{
	std_msgs::UInt8MultiArray msg;

	// resize message object to accomodate frame size
	msg.data.resize(SIZE);

	// construct message object
	memcpy(msg.data.data(), frame.data, SIZE);

	// publishing 
	ROS_INFO("Publishing...");
	pub.publish(msg);
}

/*
void UsbCamera::waitForDone()
{
	ROS_INFO("Usb camera will be wait for done");
	wait = true;
}
*/

/**
 * classify the frame really contains a trash, using gaussian distance
 * 진짜 쓰레기를 찍은 프레임인지 판단. 벡터 거리 이용
 */
Valuable UsbCamera::isValuableFrame(cv::Mat& frame)
{

	cv::Mat curFrame;
	frame.copyTo(curFrame);

	bool valuable = isValuableFrameOnInitialFrame(curFrame, initial_frame);

	if (valuable)
		return Valuable::HIGH;
	else
		return Valuable::LOW;
}

bool UsbCamera::isValuableFrameOnInitialFrame(cv::Mat& curFrame, cv::Mat& initial_frame)
{
	cv::Mat res1, res2;
	cv::subtract(curFrame, initial_frame, res1);
	cv::subtract(initial_frame, curFrame, res2);

	long distance = 0;
	for (int h = 0; h < HEIGHT; h++) {
		for (int w = 0; w < WIDTH; w++) {
			uint8_t x1 = res1.at<cv::Vec3b>(h, w)[0];
			uint8_t y1 = res1.at<cv::Vec3b>(h, w)[1];
			uint8_t z1 = res1.at<cv::Vec3b>(h, w)[2];

			uint8_t x2 = res2.at<cv::Vec3b>(h, w)[0];
			uint8_t y2 = res2.at<cv::Vec3b>(h, w)[1];
			uint8_t z2 = res2.at<cv::Vec3b>(h, w)[2];

			uint8_t x = x1 > x2 ? x1 : x2;
			uint8_t y = y1 > y2 ? y1 : y2;
			uint8_t z = z1 > z2 ? z1 : z2;

			distance += ((long)x * (long)x);
			distance += ((long)y * (long)y);
			distance += ((long)z * (long)z);
		}
	}

	distance /= (HEIGHT * WIDTH * CHANNEL);
	ROS_INFO("DISTANCE: %ld", distance);

	if (distance >= (long)THRESHOLD)
		return true;
	else
		return false;
}

/**
 * update initial_frame
 * 초기 프레임을 업데이트
 */
void UsbCamera::updateInitialFrame(cv::Mat& frame)
{
	for (int i = 0; i < SIZE; i += 1) {
		initial_frame.data[i] = (unsigned char)((float)initial_frame.data[i] * 0.9f + (float)frame.data[i] * 0.1f);
	}
}
