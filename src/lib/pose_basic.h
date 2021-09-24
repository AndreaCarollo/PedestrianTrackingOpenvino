//  _               _
// | |__  __ _  ___(_) __
// | '_ \/ _` |(_-<| |/ _|
// |_.__/\__,_|/__/|_|\__|
//

#include <vector>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "utils.h"
#ifndef POSE_BASIC
#define POSE_BASIC

using namespace cv;
using namespace std;

// body 18
const std::vector<std::pair<int, int>> posePairs = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 17}, {5, 16}};

const std::vector<std::pair<int, int>> mapIdx = {
    {31, 32}, {39, 40}, {33, 34}, {35, 36}, {41, 42}, {43, 44}, {19, 20}, {21, 22}, {23, 24}, {25, 26}, {27, 28}, {29, 30}, {47, 48}, {49, 50}, {53, 54}, {51, 52}, {55, 56}, {37, 38}, {45, 46}};

enum name_point
{
    // UPPER BODY
    NECK = 1,
    RIGHT_SHOULDER = 2,
    RIGHT_ELBOW = 3,
    RIGHT_WRIST = 4,
    LEFT_SHOULDER = 5,
    LEFT_ELBOW = 6,
    LEFT_WRIST = 7,

    // LOWER BODY
    RIGHT_HIP = 8,
    RIGHT_KNEE = 9,
    RIGHT_ANKLE = 10,
    LEFT_HIP = 11,
    LEFT_KNEE = 12,
    LEFT_ANKLE = 13,

    // FACE
    NOSE = 0,
    RIGHT_EYE = 14,
    RIGHT_EAR = 15,
    LEFT_EYE = 16,
    LEFT_EAR = 17
};

const Scalar colors_openpose[18] = {Scalar(255, 0, 0), Scalar(255, 85, 0), Scalar(255, 170, 0),
                                    Scalar(255, 255, 0), Scalar(170, 255, 0), Scalar(85, 255, 0),
                                    Scalar(0, 255, 0), Scalar(0, 255, 85), Scalar(0, 255, 170),
                                    Scalar(0, 255, 255), Scalar(0, 170, 255), Scalar(0, 85, 255),
                                    Scalar(0, 0, 255), Scalar(85, 0, 255), Scalar(170, 0, 255),
                                    Scalar(255, 0, 255), Scalar(255, 0, 170), Scalar(255, 0, 85)};

const Scalar colors_left_right[18] = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 255, 0),
                                      Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 0),
                                      Scalar(0, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 0),
                                      Scalar(255, 0, 0), Scalar(255, 0, 0), Scalar(255, 0, 0),
                                      Scalar(0, 0, 255), Scalar(0, 0, 255), Scalar(0, 255, 0),
                                      Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255)};

// keypoint stucture
struct KeyPoint_pose
{
    KeyPoint_pose(cv::Point point, float probability)
    {
        this->id = -1;
        this->point = point;
        this->probability = probability;
        this->ellipse = ellipse;
        this->covMatrix = covMatrix;
    }

    int id;
    cv::Point point;
    float probability;
    cv::RotatedRect ellipse;
    cv::Mat covMatrix;

};

// keypoint pairs
struct ValidPair
{
    ValidPair(int aId, int bId, float score)
    {
        this->aId = aId;
        this->bId = bId;
        this->score = score;
    }

    int aId;
    int bId;
    float score;
};

//   __                  _    _
//  / _| _  _  _ _   __ | |_ (_) ___  _ _   ___
// |  _|| || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

// given the heatmap, extract all the possible keypoints for pose-18-parts
void extract_all_keypoints(cv::Mat result, cv::Size targetSize,
                           vector<vector<KeyPoint_pose>> *all_keyPoints,
                           std::vector<KeyPoint_pose> *keyPointsList, float thresh = 0.10);

void extract_all_keypoints_test(cv::Mat result, cv::Size targetSize,
                                vector<vector<KeyPoint_pose>> *all_keyPoints,
                                std::vector<KeyPoint_pose> *keyPointsList, cv::Mat *img, bool plot_flag = false);

void extract_all_keypoints_cov(cv::Mat result, cv::Size targetSize,
                               vector<vector<KeyPoint_pose>> *all_keyPoints,
                               std::vector<KeyPoint_pose> *keyPointsList, cv::Mat *img, bool plot_ellipse = false, bool plot_circle = false, float thresh = 0.10);

void populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints, std::vector<cv::Point> &interpCoords);

KeyPoint_pose keypoint_from_contour(cv::Mat heatMap, std::vector<cv::Point> contour, cv::Mat *img);


// given the keypoints, extract all the valid pairs for the bodies, need the part affinity field blob
vector<cv::Mat> getValidPairs(cv::Mat paf_blob, cv::Size img_size,
                              const std::vector<std::vector<KeyPoint_pose>> detectedKeypoints,
                              std::vector<std::vector<ValidPair>> *validPairs,
                              std::set<int> *invalidPairs);

// given all the pairs of keypoints, extract the persons
void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                            const std::set<int> &invalidPairs,
                            std::vector<std::vector<int>> &personwiseKeypoints);

// plot all lines of the personwiseKeipoints
void plot_all_skeleton(cv::Mat *img, std::vector<std::vector<int>> personwiseKeypoints,
                       std::vector<KeyPoint_pose> keyPointsList, bool white = false, cv::Scalar color = cv::Scalar(0, 0, 0));

void plot_skeleton(cv::Mat *img, std::vector<int> personwiseKeypoints,
                   std::vector<KeyPoint_pose> keyPointsList, bool white = false);

void plot_skeleton_listpoint(cv::Mat *img, std::vector<Point> list_point, Scalar color = Scalar(255, 0, 0));

// get good skeleton, with a minimum of pose pairs valid
std::vector<std::vector<int>> good_skeleton_extraction(std::vector<std::vector<int>> personwiseKeypoints, int min_pairs_th = 6);

//// function for analysis

void save_heatmap(cv::Mat result, cv::Size targetSize, string path_to_save);

#endif