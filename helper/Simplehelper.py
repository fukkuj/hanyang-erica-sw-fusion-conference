import cv2
import threading as th
import os

data_path = "./data/trash_collecting/train"
labels_dict = ["nothing", "can", "glass", "paper", "plastic"]
height = 256
width = 256
index_file_name = "index.txt"

cam_index = [1]

ok = True

# image_path = os.path.join(data_path, "images")
# label_path = os.path.join(data_path, "labels")

def capture(*argv):
    global ok
    
    cap = argv[0]
    
    while ok:
        test, frame = cap.read()
        
        if test:
            cv2.imshow("Helper", frame)
            
        cv2.waitKey(10)
        

def check_for_data_dir():
    for cat in labels_dict:
        path = os.path.join(data_path, cat).replace("\\", "/")
  
        if not os.path.exists(path) or not os.path.isdir(path):
            os.mkdir(path)
            print(f"Data directory '{cat}' was created.")


def check_for_index_file():
    index_file_path = os.path.join(data_path, index_file_name).replace("\\", "/")
    
    if not os.path.exists(index_file_path) or not os.path.isfile(index_file_path):
        with open(index_file_path, "w") as index_file:
            index_file.write(str(1))
            print("Index file was created.")

def main(args):
    cap = cv2.VideoCapture(cam_index[0])
    
    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    
    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    
    if cap.isOpened() is False:
        print("cap error")

    cur_index = 1
 
    info_file_path = os.path.join(data_path, index_file_name).replace("\\", "/")
 
    check_for_index_file()
    check_for_data_dir()

    with open(info_file_path, "r") as f:
        cur_index = int(f.read())
    
    thread = th.Thread(target=capture, args=(cap,))
    thread.start()

    while True:
        try:
            print("==========================")
            print("0. NOTHING")
            print("1. can")
            print("2. glass")
            print("3. paper")
            print("4. plastic")
            ch = input("=> Label('e' to exit):")
            if ch == "e":
                break
            elif ch.isdigit():
                label = int(ch)
                if not (0 <= label <= 4):
                    continue

                _, image = cap.read()

                image = cv2.resize(image, dsize=(height, width))
                cv2.imwrite(os.path.join(data_path, "{}/%08d.jpg".format(labels_dict[label])).replace("\\", "/") % cur_index, image)
                # label_file.write("{0} {1}/%08d.jpg\n".format(label, labels_dict[label]) % cur_index)

                cur_index += 1
        except:
            break

    with open(os.path.join(data_path, index_file_name).replace("\\", "/"), "w") as f:
        f.write(str(cur_index))

    # label_file.close()
    
    global ok
    ok = False
    
    thread.join()
    
    cap.release()

if __name__ == "__main__":
    import sys
    main(sys.argv)