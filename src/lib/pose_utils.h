#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <string>
#include <math.h>

#include "pose_basic.h"
#include "Hungarian.h"

#ifndef POSE_UTILS
#define POSE_UTILS

using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

//  _                          _        _                _    _  _
// | |__ ___  _  _  _ __  ___ (_) _ _  | |_  ___   _  _ | |_ (_)| | ___
// | / // -_)| || || '_ \/ _ \| || ' \ |  _|(_-<  | || ||  _|| || |(_-<
// |_\_\\___| \_, || .__/\___/|_||_||_| \__|/__/   \_,_| \__||_||_|/__/
//            |__/ |_|
//

// Returns True if the 3 points A,
// B and C are listed in a counterclockwise order
// ie if the slope of the line AB is less than the slope of AC
// https : //bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
bool ccw(cv::Point A, cv::Point B, cv::Point C);

// Return true if line segments AB and CD intersect
// https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
bool intersect(cv::Point A, cv::Point B, cv::Point C, cv::Point D);

// Calculate the angle between segment(A,B) and segment (B,C)
float angle(cv::Point A, cv::Point B, cv::Point C);

// Calculate the angle between segment(A,B) and vertical axe
float vertical_angle(cv::Point A, cv::Point B);

// Calculate the square of the distance between points A and B
float sq_distance(cv::Point A, cv::Point B);

// Calculate the distance between points A and B
float distance(cv::Point A, cv::Point B);

long double distanceMahalanobis(cv::Point A, cv::Point B_cov, cv::Mat covMat);

//  ___                  _    _
// | __| _  _  _ _   __ | |_ (_) ___  _ _   ___
// | _| | || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

cv::RotatedRect getErrorEllipse(float chisquare_val, cv::Point2f mean, cv::Mat covmat);

cv::RotatedRect generate_cov_ellipse(cv::Mat heatMap, vector<Point> contour,cv::Mat *covMat);

// interactive functions

void text_select_key(int number);

// get user clicked point in a window
void mouse_callback(int event, int x, int y, int flag, void *param);

// get user clicked point in a window
void get_point_user(Mat *img_get_point, vector<Point> *clicked_point);

// generate bounding box for a pose (with pointer to list of keypoints_pose)
cv::Rect get_bounding_rect_for_pose(std::vector<int> pose, std::vector<KeyPoint_pose> keyPointsList);

// generate bounding box for a pose (list of points)
cv::Rect get_bounding_rect_for_listpose(std::vector<cv::Point> pose);

vector<int> generate_match_poses(std::vector<std::vector<int>> pose_1, std::vector<KeyPoint_pose> keyPointsList_1,
                                 std::vector<std::vector<int>> pose_2, std::vector<KeyPoint_pose> keyPointsList_2,
                                 Mat *display_img,
                                 bool print_box = false, bool print_assignment = true);

void save_point_and_distances(vector<int> assignment, vector<int> *euclidean_distances, vector<int> *mahalanobis_distances,
                              vector<vector<int>> poses_peak, vector<KeyPoint_pose> keyPointsList_peak,
                              vector<vector<int>> poses_cov, vector<KeyPoint_pose> keyPointsList_cov,
                              string folder, bool save_to_file = false);

#endif