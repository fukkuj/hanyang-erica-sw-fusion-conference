#ifndef __USB_CAMERA_HPP__
#define __USB_CAMERA_HPP

#include "ros/ros.h"
#include "std_srvs/SetBool.h"
#include "opencv2/opencv.hpp"

/**
 * Instruction:
 * 
 * if the trash is in the classifier box, camera start capturing the trash.
 * if enough picture is collected, camera is stopped and send trash image to other node.
 * 
 * if other node send message which the trash is processed, camera should be resumed.
 * 
 * this class contains all about above functionalities.
 * 
 * in detail, please see .cpp file
 * 
 * 작동 방식:
 * 
 * 만약, 쓰레기가 박스에 들어오면, 카메라는 사진을 찍기 시작한다.
 * 사진이 충분히 모이면, 카메라는 작동을 중지하고 이미지를 다른 노드에게 전달한다.
 * 만약, 쓰레기 처리가 끝나고 다른 노드로부터 완료 신호를 받으면, 카메라를 재개한다.
 * 
 * 이 클래스는 위와 같은 기능을 포함한다.
 * 
 * 상세한 메소드별 설명은 .cpp파일을 보길 바란다.
 */
class UsbCamera {
public:
	UsbCamera(ros::NodeHandle& handle);
	virtual ~UsbCamera();

	void init();
	void finalize();
	void compute();
	//void resume();
	//void waitForDone();
	//bool isWaiting();

private:
	bool stop;									// ??
	//bool wait;
	bool isReady;								// is ready to process next trash?

	ros::NodeHandle nh;							// to create ros functionality
	ros::Publisher pub;							// to send message/image to other node.
	ros::ServiceServer cam_done_serv_server;	// to receive ready message
	cv::VideoCapture* cap1, cap2;				// camera objects
	cv::Mat initial_frame1, initial_frame2;		// initial frames. this is used to detect whether a trash is really in the box.

	bool readyService(std_srvs::SetBool::Request& req,
			  std_srvs::SetBool::Response& res);
	bool isValuableFrame(cv::Mat& frame);
	void publish(cv::Mat& frame);
	void updateInitialFrame(cv::Mat& frame1, cv::Mat& frame2);
};

#endif
