
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

    print("Start server!")
    server = Server()
    server.open(host, port)

    ai = AI()
    ai.build()

    # debug
    #t = th.Thread(target=image_show)
    #t.start()

    try:
        while True:
            image_arr = np.zeros((1, num_step, image_channel*2, HEIGHT, WIDTH), dtype=np.float32)
            cnt = 0
            index = 0
            
            while cnt < num_step:

                image_bgr = server.wait_for_image()
                if image_bgr is None:
                    continue

                # print(image_bgr.shape)

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"./test/{index}.jpg", image_bgr)
                # image_rgb = cv2.cvtColor(cv2.imread(f"./test/{index}.jpg"), cv2.COLOR_BGR2RGB)
                index += 1

                image_arr[0, cnt] = image_rgb.transpose(2, 0, 1).astype(np.float32)
                cnt += 1

                image_rgb = None
                image_bgr = None

            result = ai.predict(image_arr)
            print("Result: {}".format(trash_map[result]))

            server.send_result(result)

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


if __name__ == "__main__":
    main(sys.argv)
