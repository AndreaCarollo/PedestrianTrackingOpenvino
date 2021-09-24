// Double inclusion guard
#ifndef RS_UTILITY
#define RS_UTILITY

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

// Convert the rs color frame into a openCV Mat
cv::Mat frame_to_mat(const rs2::frame& f, bool BGR_RGB);

rs2::config config_pipe(bool flag_small = true);

#endif