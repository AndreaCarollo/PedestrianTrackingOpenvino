#include "Tracker_by_detection.hpp"

void do_tracking(cv::Mat *img, cv::Mat *display_img,
                 Tracker_by_detection *UserToTrack, parameters *param,
                 cv::dnn::Net *network_Pedestrian, cv::dnn::Net *network_faces,
                 cv::dnn::Net *network_Feature_Body);