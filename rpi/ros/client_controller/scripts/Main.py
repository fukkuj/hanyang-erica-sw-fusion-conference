#!/usr/bin/env python
#-*- coding:utf-8 -*-

import rospy
from Client import Client
from ImageSubscriber import ImageSubscriber
from MotorPublisher import MotorPublisher
import sys
import threading as th
import time

HOST =  "35.229.136.239"
#HOST =  "35.236.168.202"
#HOST = "34.80.12.180"
#HOST = "192.168.43.150"
PORT = 13333
NUM_STEP = 8

BOX_MOTOR = 0 # Move trash box
SUPPORT_MOTOR = 1 # open or close gate

result = None
ok = True

def recv_result(client):
    global result
    global ok

    while ok:
        result = client.recv_result()
        rospy.loginfo("Result: " + str(result))


def motor_control(result, control_queue):

    if result == -1:
        return
    elif result == 0:
        control_queue.append([BOX_MOTOR, 1, 1800])
        control_queue.append([SUPPORT_MOTOR, 0, 1800])
        control_queue.append([SUPPORT_MOTOR, 1, 1810])
        control_queue.append([BOX_MOTOR, 0, 1780])
    elif result == 1:
        control_queue.append([BOX_MOTOR, 1, 850])
        control_queue.append([SUPPORT_MOTOR, 0, 1800])
        control_queue.append([SUPPORT_MOTOR, 1, 1810])
        control_queue.append([BOX_MOTOR, 0, 860])
    elif result == 2:
        control_queue.append([BOX_MOTOR, 0, 850])
        control_queue.append([SUPPORT_MOTOR, 0, 1800])
        control_queue.append([SUPPORT_MOTOR, 1, 1810])
        control_queue.append([BOX_MOTOR, 1, 820])
    elif result == 3:
        control_queue.append([BOX_MOTOR, 0, 1820])
        control_queue.append([SUPPORT_MOTOR, 0, 1800])
        control_queue.append([SUPPORT_MOTOR, 1, 1810])
        control_queue.append([BOX_MOTOR, 1, 1755])
    else:
        rospy.loginfo("INVALID result " + str(result))
        return

def main(argv):
    global result
    global ok

    rospy.init_node("client_controller", anonymous=True)

    client = Client()
    if client.connect(HOST, PORT) is False:
        return

    time.sleep(5)

    image_sub = ImageSubscriber()
    motor_pub = MotorPublisher()

    rate = rospy.Rate(20)

    t = th.Thread(target=recv_result, args=(client,))
    t.start()

    cnt = 0;
    motor_control_queue = []
    wait_for_result = False
    is_processing = False

    motor_pub.cam_ready()

    while not rospy.is_shutdown():
        if wait_for_result:
            if result is not None:
                motor_control(result, motor_control_queue)
                result = None
                wait_for_result = False
                is_processing = True

            rate.sleep()
            continue

        elif is_processing:
            if len(motor_control_queue) > 0 and motor_pub.is_ready():
                motor_id, direction, distance = motor_control_queue.pop(0)
                motor_pub.publish(motor_id, direction, distance)
            elif len(motor_control_queue) == 0 and motor_pub.is_ready():
                cnt = 0
                image_sub.cnt = 0
                is_processing = False

                client.ready = True
                image_sub.ready = True
                motor_pub.cam_ready()
                
            rate.sleep()
            continue

        images = image_sub.get_image()
        if images is None:
            rate.sleep();
            continue

        client.send_image(images[0])
        client.send_image(images[1])
        cnt += 1

        if cnt == NUM_STEP:
            wait_for_result = True
            client.ready = False

        rate.sleep()

    ok = False
    t.join()

if __name__ == "__main__":
    main(sys.argv)
