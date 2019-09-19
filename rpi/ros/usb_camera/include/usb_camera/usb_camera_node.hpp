#ifndef __USB_CAMERA_HPP__
#define __USB_CAMERA_HPP__

#include "std_srvs/SetBool.h"

#define TIME_INTERVAL 250                   // Interval between camera captures
#define HEIGHT 128                           // height of image
#define WIDTH 128                            // width of image
#define CHANNEL 3                           // channel of image BGR = 3
#define SIZE (HEIGHT * WIDTH * CHANNEL)     // size of image

#define THRESHOLD 100                       // valuability threshold

#endif
