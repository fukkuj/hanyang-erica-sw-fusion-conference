# 인공지능 분리수거 쓰레기통
## 개요
길거리를 걷다 보면, 분리수거가 제대로 되어 있지 않은 쓰레기통이 많다. 보통 분리수거를 하지 않는 이유를 보면, 귀찮아서, 또는 정확히 어디로 분류해야 하는지 몰라서 인 경우가 많다. 이런 이유로 분리수거가 제대로 되어 있지 않고, 이는 사회적 문제가 되고 있다. 우리는 쓰레기를 분류하는 과정을 자동화하는 시스템을 개발하고자 했다.
단순히 귀찮거나 정확한 카테고리를 몰라서 분리수거를 하지 않는 경우도 있지만, 시각에 불편함을 겪고 계신 분들처럼 분리수거를 제대로 하기 어려운 경우가 있을 것이다. 우리는 공공장소에 도움이 될 뿐 아니라 몸이 불편하신 분들에게도 도움이 되었으면 한다.
우리 프로젝트는 라즈베리파이가 쓰레기통의 각 모듈을 제어하고 있고, 원격 서버에서 인공지능을 구동하는 구조이다. 라즈베리파이 위에 올라가는 소프트웨어는 ROS(Robot Operating Systems) 플랫폼 위에서 구동되도록 작성되었으며, 여러개의 노드(ROS의 프로세스)가 유기적으로 통신하면서 동작하게 된다. 인공지능의 경우, 딥러닝을 사용했으며, 라즈베리파이에서 구동하기보단 원격 서버에서 구동한다. 원격 서버는 GCP(Google Cloud Platform)의 compute engine을 사용했다. 딥러닝 모델은 PyTorch로 구현했다.

## 작동 과정
쓰레기를 집어 넣으면 딥러닝이 해당 쓰레기를 캔, 유리, 종이, 플라스틱 중 하나로 판별하게 된다. 판별된 결과를 바탕으로 라즈베리 파이가 스테핑 모터를 조종해서 쓰레기를 알맞은 분리수거함에 수거하게 된다. 작동 과정은 다음과 같다.

![image](https://user-images.githubusercontent.com/26874750/66266151-04ff8280-e85c-11e9-956f-d17cd98b18d9.png)

1. 사람이 쓰레기를 버리면 카메라로 쓰레기를 촬영한다.
2. 라즈베리파이에서 쓰레기 이미지를 원격 서버로 보낸다.
3. 원격 서버에서 딥러닝 알고리즘을 통해 쓰레기를 판별한다.
4. 판별한 결과를 라즈베리파이로 돌려준다.
5. 라즈베리파이에서 스테핑 모터로 내릴 명령을 생성한다.
6. 스테핑 모터로 명령을 내린다.

## 요구 사항
**준비 모듈**: 라즈베리파이 3B+, 클라우드 플랫폼의 가상 서버, 카메라 2개, 초음파센서 4개, 스테퍼 모터 4개

### 라즈베리파이 셋업
1. Raspbian Stretch 운영체제를 라즈베리파이 위에 설치한다. (참고: https://www.raspberrypi.org/documentation/installation/installing-images/README.md)
2. ROS melodic를 라즈베리파이에 설치한다. (참고: http://wiki.ros.org/ROSberryPi/Installing%20ROS%20Kinetic%20on%20the%20Raspberry%20Pi)
3. ROS workspace인 ```~/catkin_ws/src```안에서 ```catkin_create_pkg client_controller```, ```catkin_create_pkg usb_camera```, ```catkin_create_pkg stepper_motor```, ```catkin_create_pkg ir_controller```를 실행한다(순서는 상관없음). 그리고 생성된 네 개의 폴더에 프로젝트의```rpi/ros/``` 하위 폴더를 모두 복사한다.

### 원격 서버 셋업(GCP 기준)
1. Compute engine을 만들고 ssh 세팅을 한다. 이 과정은 GCP 이용법이므로 생략한다. 다만, OS 이미지는 pytorch, cuda-10.0 버전을 선택하자.
2. Compute engine을 실행하고 파이썬 가상환경을 만든다. ```conda create -n torchenv numpy opencv pillow matplotlib jupyterlab```
3. Pytorch를 가상환경에 설치한다. ```conda activate torchenv && conda install pytorch -c pytorch```

## 사용법
하드웨어가 세팅되어 있다면,
1. GCP에서 ```run_server.py``` 실행
2. 라즈베리파이에서 ```roslaunch ai_recycling_bins launch.launch```를 실행한다.

## DEMO
https://www.notion.so/wayexists/SW-ICT-Conference-35aa01d66e5b43b5971091dc787eb39a

## 참조
초음파 센서 코드: https://github.com/OmarAflak/HC-SR04-Raspberry-Pi-C-
