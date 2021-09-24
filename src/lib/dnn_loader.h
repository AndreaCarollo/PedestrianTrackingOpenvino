#include <opencv2/dnn.hpp>
#include <string>

#ifndef DNN_LOADER
#define DNN_LOADER

cv::dnn::Net load_detection_people();

cv::dnn::Net load_detection_faces();

cv::dnn::Net load_reid_people();

cv::dnn::Net load_reid_faces();


const std::string PAF_blobName = "Mconv7_stage2_L1";
const std::string HM_blobName = "Mconv7_stage2_L2";
const std::vector<std::string> outBlobNames = {HM_blobName, PAF_blobName};

void load_net_pose(cv::dnn::Net *network_pose);

#endif
