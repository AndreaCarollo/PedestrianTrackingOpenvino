//                                  _                   _
//  _ __  ___  ___ ___    ___ __ __| |_  _ _  __ _  __ | |_  ___  _ _
// | '_ \/ _ \(_-</ -_)  / -_)\ \ /|  _|| '_|/ _` |/ _||  _|/ _ \| '_|
// | .__/\___//__/\___|  \___|/_\_\ \__||_|  \__,_|\__| \__|\___/|_|
// |_|
//

#include "pose_basic.h"
#include "pose_utils.h"
#include "dnn_loader.h"

const std::vector<int> nullpose = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

// from the image get all the human poses on the frame
std::vector<std::vector<int>> get_poses(cv::Mat *img, cv::dnn::Net *network_pose,
                                        std::vector<KeyPoint_pose> *keyPointsList,
                                        bool filter_poses = false, int min_keypoint = 6, float scale_blob = 2, float thresh = 0.10);

std::vector<std::vector<int>> get_poses_cov(cv::Mat *img, cv::dnn::Net *network_pose,
                                            std::vector<KeyPoint_pose> *keyPointsList,
                                            bool filter_poses, int min_keypoint = 6, float scale_blob = 2, float thresh=0.10);

std::vector<int> parse_poses(cv::Rect *ROI, std::vector<KeyPoint_pose> keyPointsList,
                             std::vector<std::vector<int>> *poses, cv::Mat *img);