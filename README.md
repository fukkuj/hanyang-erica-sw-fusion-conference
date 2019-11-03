# 인공지능 분리수거 쓰레기통 2

캡스톤 프로젝트의 업그레이드 버전입니다.

### 구성원

- 이재영(한양대 ERICA SW학부, 4학년)
- 박병우(한양대 ERICA 로봇공학과, 4학년)
- 문현준(한양대 ERICA SW학부, 4학년)
- 박재선(한양대 ERICA SW학부, 2학년)
- 이윤지(한양대 ERICA ICT융합학부, 2학년)

## 개요
길거리를 걷다 보면, 분리수거가 제대로 되어 있지 않은 쓰레기통이 많다. 보통 분리수거를 하지 않는 이유를 보면, 귀찮아서, 또는 정확히 어디로 분류해야 하는지 몰라서 인 경우가 많다. 이런 이유로 분리수거가 제대로 되어 있지 않고, 이는 사회적 문제가 되고 있다. 우리는 쓰레기를 분류하는 과정을 자동화하는 시스템을 개발하고자 했다.
단순히 귀찮거나 정확한 카테고리를 몰라서 분리수거를 하지 않는 경우도 있지만, 시각에 불편함을 겪고 계신 분들처럼 분리수거를 제대로 하기 어려운 경우가 있을 것이다. 우리는 공공장소에 도움이 될 뿐 아니라 몸이 불편하신 분들에게도 도움이 되었으면 한다.
우리 프로젝트는 라즈베리파이가 쓰레기통의 각 모듈을 제어하고 있고, 원격 서버에서 인공지능을 구동하는 구조이다. 라즈베리파이 위에 올라가는 소프트웨어는 ROS(Robot Operating Systems) 플랫폼 위에서 구동되도록 작성되었으며, 여러개의 노드(ROS의 프로세스)가 유기적으로 통신하면서 동작하게 된다. 인공지능의 경우, 딥러닝을 사용했으며, 라즈베리파이에서 구동하기보단 원격 서버에서 구동한다. 원격 서버는 GCP(Google Cloud Platform)의 compute engine을 사용했다. 딥러닝 모델은 PyTorch로 구현했다.


# How It Works

---

모든 실행 전에, 라즈베리 파이와 그를 제어할 노트북은 모두 jylee 핫스팟에 연결되어 있어야 합니다.

1. 라즈베리 전원 어댑터를 먼저 콘센트에 연결하고, SMPS 전원을 콘센트에 연결합니다. 순서가 중요한건 아니지만, 라즈베리가 꺼진 상태에서 SMPS를 먼저 연결하면, 모터에서 잠시동안 이상한 소리가 날 수 있습니다. 라즈베리가 켜지면 소리는 사라집니다.
2. GCP에 있는 deep learning 서버와 mobile server를 킵니다(이건 팀장한테 요청합니다).

        ### 1번째 shell ###
        
        # activate virtual environment
        conda activate torchenv
        
        # change working directory
        cd ~/projects/sw-conference
        
        # run ai server
        python run_ai_server.py

        ### 2번째 shell ###
        
        # change working directory
        cd ~/projects/sw-conference
        
        # run mobile server
        java -jar mobileserver.jar

3. 첫번째 Raspberry Pi 에서 ROS node들을 실행시킵니다.

        ### ssh pi@192.168.43.89 로 접속 passwd: qwer1234
        
        roslaunch ai_recycling_bins launch.launch

4. 두번째 Raspberry Pi에서 초음파 센서를 실행시킵니다. (2번 과정에서 쓰레기통이 접속되었다는 출력이 있었으면 할 필요 없음. 아마 자동으로 실행되어 할 필요가 없을 겁니다)

        ### ssh pi@192.168.43.129 로 접속 passwd: qwer
        
        python hr_sensor/hcsr.py

5. 모바일 어플리케이션을 실행합니다(선택사항)

# Control Flow

전체적인 흐름은 다음과 같습니다.
![image](https://user-images.githubusercontent.com/26874750/68081237-344ee280-fe4e-11e9-98c4-c418eb29a035.png)

## 작동 과정
쓰레기를 집어 넣으면 딥러닝이 해당 쓰레기를 캔, 유리, 종이, 플라스틱 중 하나로 판별하게 된다. 판별된 결과를 바탕으로 라즈베리 파이가 스테핑 모터를 조종해서 쓰레기를 알맞은 분리수거함에 수거하게 된다. 작동 과정은 다음과 같다.

![image](https://user-images.githubusercontent.com/26874750/66266151-04ff8280-e85c-11e9-956f-d17cd98b18d9.png)

1. 사람이 쓰레기를 버리면 카메라로 쓰레기를 촬영한다.
2. 라즈베리파이에서 쓰레기 이미지를 원격 서버로 보낸다.
3. 원격 서버에서 딥러닝 알고리즘을 통해 쓰레기를 판별한다.
4. 판별한 결과를 라즈베리파이로 돌려준다.
5. 라즈베리파이에서 스테핑 모터로 내릴 명령을 생성한다.
6. 스테핑 모터로 명령을 내린다.


# Neural Network Architectures

저희 프로젝트에서는 2개의 신경망을 구축했으며, 하나는 쓰레기가 있는지 없는지를 판단하고 다른 하나는 쓰레기를 4가지 카테고리로 분류하는 역할을 합니다.

## Trash Detector

다음은 첫 번째 신경망으로, 쓰레기인지, 아닌지 binary classification을 수행합니다.

![image](https://user-images.githubusercontent.com/26874750/68081261-69f3cb80-fe4e-11e9-9af6-79fbdee333c7.png)

Feature를 추출하기 위해 사전 학습된 VGG11의 convolution 파트만 가져와서 저희들의 데이터로 재학습(fine-tuning) 시켰습니다. 그리고, 새로운 fully connected 레이어를 연결해서 binary classifier 구조를 만든 후, VGG11 파트의 weights값들을 고정하고 fully connected 부분을 학습시켰습니다.

저희 학습 데이터 기준 99%정도의 정확도, 검증 데이터 기준 99% 정도의 정확도가 나왔지만, 검증 데이터에 대한 high variance를 의심해 봐야 할 것 같습니다.

## Trash Classifier

Trash detector를 거쳐서 진짜 쓰레기라고 판단되면, Trash classifier로 이미지가 넘어옵니다. trash classifier는 쓰레기를 4가지 카테고리로 분류하게 됩니다.

![image](https://user-images.githubusercontent.com/26874750/68081267-79731480-fe4e-11e9-8cd4-2222c8833e23.png)

Trash classifier역시, VGG11의 feature부분만 가져와서 feature를 추출하는 역할로 사용했습니다. 이 부분을 저희 데이터로 재학습(fine-tuning)시킨 후, 다시 새로운 classifier를 붙인 형태인데요, 이 네트워크의 특징은 이미지 16장이 동시에 들어간다는 것입니다. 카메라 2대로 각각 8장씩 찍는데, 이 16장의 이미지가 trash classifier의 한 번의 입력으로 들어갑니다.

이 16장의 이미지는 각각 VGG11를 통과한 후, 일렬로 concatenate하게 됩니다. 이 concatenate시킨 벡터를 1D-convolution 네트워크에 통과시키고, 마지막으로 fully connected 네트워크에 통과시킵니다.

이 네트워크로 학습 데이터 기준 약 99%, 검증 데이터 기준 약 98% 정도 나왔습니다. 역시, 검증 데이터에 대한 high variance가 의심되나, 테스트 데이터를 따로 두지 않았습니다.


## DEMO
https://www.notion.so/wayexists/SW-ICT-Conference-35aa01d66e5b43b5971091dc787eb39a

## 참조
초음파 센서 코드: https://github.com/OmarAflak/HC-SR04-Raspberry-Pi-C-
