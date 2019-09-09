
import sys
import numpy as np
import cv2
import threading as th
from server.Server import Server

host = "0.0.0.0"
#host = "192.168.137.1"
port = 13333
num_step = 8
num_classes = 4
image_width = 96
image_height = 96
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
            image_arr = np.zeros((num_step, image_channel*2, image_height, image_width))
            cnt = 0
            index = 0
            
            while cnt < num_step:

                image_bgr1 = server.wait_for_image()
                image_bgr2 = server.wait_for_image()
                if image_bgr1 is None or image_bgr2 is None:
                    continue

                # print(image_bgr.shape)

                image_rgb1 = cv2.cvtColor(image_bgr1, cv2.COLOR_BGR2RGB)
                image_rgb2 = cv2.cvtColor(image_bgr2, cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"./test/{index}_1.jpg", image_bgr1)
                cv2.imwrite(f"./test/{index}_2.jpg", image_bgr2)
                # image_rgb = cv2.cvtColor(cv2.imread(f"./test/{index}.jpg"), cv2.COLOR_BGR2RGB)
                index += 1

                image_arr[cnt, :image_channel] = image_rgb1.transpose(2, 0, 1)
                image_arr[cnt, image_channel:] = image_rgb2.transpose(2, 0, 1)
                cnt += 1

                image_rgb1 = None
                image_rgb2 = None
                image_bgr1 = None
                image_bgr2 = None

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

    server.close()
    ok = False
    # t.join()


if __name__ == "__main__":
    main(sys.argv)
