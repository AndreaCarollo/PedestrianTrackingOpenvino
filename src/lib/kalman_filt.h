
//  _  __       _                       ___  _  _  _             
// | |/ / __ _ | | _ __   __ _  _ _    | __|(_)| || |_  ___  _ _ 
// | ' < / _` || || '  \ / _` || ' \   | _| | || ||  _|/ -_)| '_|
// |_|\_\\__,_||_||_|_|_|\__,_||_||_|  |_|  |_||_| \__|\___||_|  
// 

#ifndef KALMAN
#define KALMAN

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <armadillo>
#include <iostream>

class kalman_filt
{
private:
    /* data */
public:
    bool found = false;
    int notFoundCount = 0;
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    cv::Mat state, meas;
    cv::KalmanFilter kf;
    cv::Mat m;

    std::chrono::time_point<std::chrono::system_clock> tick = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tick_prev = std::chrono::system_clock::now();
    float DT = 0;

    cv::Rect predRect, postRect;
    cv::Point center, postCenter;

    void init();

    // take dt in seconds
    void get_time();
    // Dinamica libera del sistema x+ = A x
    void generateA(float dt);
    // estimation of acceleration from historic data
    void estimate_acc_input();
    // do a step of kalman filter
    void prediction(cv::Mat *display_img, bool plot = false);

    void plot_prediction(cv::Mat *display_img);
    void plot_post(cv::Mat *display_img);

    // take measures from the real history position
    void get_measure(std::vector<cv::Point> storic_positions);
    // posterior update of the estimate states values
    void update(cv::Rect measure_box, bool state_track, cv::Mat *display_img, bool plot = false);
    // save the estimate states keeping only \history_max_length values
    void update_history();
};

#endif