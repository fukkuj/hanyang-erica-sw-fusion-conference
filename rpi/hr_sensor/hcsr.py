import RPi.GPIO as GPIO
import time
from socket import *
from threading import Thread
from signal import signal, SIGINT


trig1 = 19
echo1 = 21
trig2 = 13   
echo2 = 20
trig3 = 16
echo3 = 26
trig4 = 12
echo4 = 6

trigs = [trig1, trig4, trig3, trig2]
echos = [echo1, echo4, echo3, echo2]

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(trig1, GPIO.OUT)
GPIO.setup(echo1, GPIO.IN)
GPIO.setup(trig2, GPIO.OUT)
GPIO.setup(echo2, GPIO.IN)
GPIO.setup(trig3, GPIO.OUT)
GPIO.setup(echo3, GPIO.IN)
GPIO.setup(trig4, GPIO.OUT)
GPIO.setup(echo4, GPIO.IN)

host = "192.168.43.150"
#host = "35.229.136.239"
port = 13345
interrupted = False
interval = 1

values = [0, 0, 0, 0]

def get_value(trig, echo):
    try :
        GPIO.output(trig,False)
        time.sleep(1)
        GPIO.output(trig,True)
        time.sleep(0.00001)
        GPIO.output(trig,False)
        while GPIO.input(echo) == 0 :
            pulse_start = time.time()
            #print"pulse_start num: " , pulse_start
            #print "pulse_start : %d", pulse_start
        
        while GPIO.input(echo) == 1:
            pulse_end = time.time()
            #print"pulse_end : %d", pulse_end

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17000
        distance = round(distance, 2)

        print "distance: ", distance, "cm"

        ratio = distance - 10
        if ratio > 40: ratio = 40
        elif ratio < 0: ratio = 0
        return (40 - ratio) / 40

    except Keyboardinterrupt :
        GPIO.cleanup()
        return None

def connect_to_server(host, port):
    client = socket(AF_INET, SOCK_STREAM)
    client.connect((host, port))

    client.send("1\n".encode('utf-8'))

    th = Thread(target=client_thread, args=(client,))
    th.start()


def client_thread(*argv):
    client = argv[0]
   
    try:
        while not interrupted:
            data = client.recv(8)
            print "%s" %data.decode('utf-8')

            data = u"%d %d %d %d\n" %(int(values[0]), int(values[1]), int(values[2]), int(values[3]))
            client.send(data.encode('utf-8'))

    except:
        print "CANNOT connect to server!"

    client.close()


def handler(signo, frame):
    global interrupted
    interrupted = True


def main():
    global values

    signal(SIGINT, handler)

    connect_to_server(host, port)

    prev_time = time.time()

    while not interrupted:

        if time.time() - prev_time > interval:
            for i in xrange(4):
                values[i] = get_value(trigs[i], echos[i]) * 100


if __name__ == "__main__":
    main()

