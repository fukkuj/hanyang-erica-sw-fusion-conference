
import sys
import numpy as np
import cv2
import threading as th
from ai_server.Server import Server
from env import *

host = "0.0.0.0"
#host = "192.168.137.1"
port = 13333
num_step = 8
num_classes = 4
image_channel = 3

from ai.AI import AI

image_bgr = None
ok = True
interrupted = False
server = None

trash_map = ["can", "glass", "paper", "plastic", "nothing"]


def image_show():
    global image_bgr
    global ok

    while ok:
        if image_bgr is not None:
            cv2.imshow("test", cv2.resize(image_bgr, dsize=(240, 240)))
            cv2.waitKey(1000 // 60)


def main(args):
    global image_bgr
    global ok
    global server

    print("Start server!")
    server = Server()
    server.open(host, port)

    ai = AI()
    ai.build()

    # debug
    #t = th.Thread(target=image_show)
    #t.start()

    burn_in = 0

    try:
        while not interrupted:
            image_arr = np.zeros((1, num_step, image_channel*2, HEIGHT, WIDTH), dtype=np.float32)
            cnt = 0
            index = 0
            
            while cnt < num_step:

                image_bgr1 = server.wait_for_image()
                image_bgr2 = server.wait_for_image()
                if image_bgr1 is None or image_bgr2 is None:
                    continue

                if burn_in < num_step:
                    cnt += 1
                    burn_in += 1
                    continue

                # print(image_bgr.shape)

                image_rgb1 = cv2.cvtColor(image_bgr1, cv2.COLOR_BGR2RGB)
                image_rgb2 = cv2.cvtColor(image_bgr2, cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"./test/{index}_1.jpg", image_bgr1)
                cv2.imwrite(f"./test/{index}_2.jpg", image_bgr2)
                # image_rgb = cv2.cvtColor(cv2.imread(f"./test/{index}.jpg"), cv2.COLOR_BGR2RGB)
                index += 1

                image_arr[0, cnt, :3] = image_rgb1.transpose(2, 0, 1).astype(np.float32)
                image_arr[0, cnt, 3:] = image_rgb2.transpose(2, 0, 1).astype(np.float32)
                cnt += 1

                image_rgb1 = None
                image_rgb2 = None
                image_bgr1 = None
                image_bgr2 = None
                burn_in += 1

            if burn_in == 2*num_step:
                result = ai.predict(image_arr)
                print("Result: {}".format(trash_map[result]))

                server.send_result(result)
                burn_in = 0
            elif burn_in == num_step:
                server.send_result(-1)

    except KeyboardInterrupt as e:
        print(e)
        print("Keyboard Interrupted.")

    except ValueError as e:
        print(e)
        print("Exception occurs. Server shutdown.")

    except TypeError as e:
        print(e)
        print("Exception occurs. Server shutdown.")
    
    except:
        pass

    server.close()
    ok = False
    # t.join()


def handler(signo, frame):
    interrupted = True
    server.close()


if __name__ == "__main__":
    main(sys.argv)
