//  ______    _____    __       __        _____   ___          ___  __      __   ______
// |   ___|  /  _  \  |  |     |  |      /  _  \  \  \        /  / |   \  /   | |  ____| 
// |  |__   |  | |  | |  |     |  |     |  | |  |  \  \  /\  /  /  | |\ \/ /| | | |___
// |   __|  |  | |  | |  |     |  |     |  | |  |   \  \/  \/  /   | | \__/ | | |  ___|
// |  |     |  |_|  | |  |___  |  |___  |  |_|  |    \   /\   /    | |      | | | |____
// |__|      \_____/  |______| |______|  \_____/      \_/  \_/     |_|      |_| |______|
//
// This is the common project header file
// Declare and define here types and structures that are of common usage


// Double inclusion guard
#ifndef FOLLOWME_H
#define FOLLOWME_H

// System headers
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <chrono> // for time measurement
#include <ctime>

// OpenCV Library
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>

// Realsense Library
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_frame.hpp>


//>>>>>>>>>>>>>>>>>>>> Project libraries:
// realsense utilities
#include "rs_utility.h"

// load dnn nets
#include "dnn_loader.h"
// human pose utilities
#include "pose_extractor.h"
#include "pose_gesture.h"
// libraries for tracking people
#include "utils.h"
#include "Tracking.h"
// #include "Tracker_by_detection.hpp"
#include "kalman_filt.h"

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#endif
