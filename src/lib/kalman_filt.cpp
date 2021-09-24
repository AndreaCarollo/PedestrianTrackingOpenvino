
//  _  __       _                       ___  _  _  _
// | |/ / __ _ | | _ __   __ _  _ _    | __|(_)| || |_  ___  _ _
// | ' < / _` || || '  \ / _` || ' \   | _| | || ||  _|/ -_)| '_|
// |_|\_\\__,_||_||_|_|_|\__,_||_||_|  |_|  |_||_| \__|\___||_|
//

#include "kalman_filt.h"

using namespace std;

void kalman_filt::init()
{
    kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
    state = cv::Mat(stateSize, 1, CV_32F); // [x,y,v_x,v_y,w,h]
    meas = cv::Mat(measSize, 1, CV_32F);   // [z_x,z_y,z_w,z_h]

    cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // kf.processNoiseCov.at<float>(0) = 1e-2;
    // kf.processNoiseCov.at<float>(7) = 1e-2;
    // kf.processNoiseCov.at<float>(14) = 5.0f;
    // kf.processNoiseCov.at<float>(21) = 5.0f;
    // kf.processNoiseCov.at<float>(28) = 1e-2;
    // kf.processNoiseCov.at<float>(35) = 1e-2;

    kf.processNoiseCov.at<float>(0) = 1.0f;
    kf.processNoiseCov.at<float>(7) = 1.0f;
    kf.processNoiseCov.at<float>(14) = 2.0f;
    kf.processNoiseCov.at<float>(21) = 2.0f;
    kf.processNoiseCov.at<float>(28) = 1.0f;
    kf.processNoiseCov.at<float>(35) = 1.0f;

    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
}

void kalman_filt::prediction(cv::Mat *display_img, bool plot)
{
    if (found)
    {
        // >>>> Matrix A
        kf.transitionMatrix.at<float>(2) = DT;
        kf.transitionMatrix.at<float>(9) = DT;
        // <<<< Matrix A
        // std::cout << "dT: " << dt_kalman << std::endl;

        state = kf.predict();
        predRect.width = state.at<float>(4);
        predRect.height = state.at<float>(5);
        predRect.x = state.at<float>(0) - predRect.width / 2;
        predRect.y = state.at<float>(1) - predRect.height / 2;

        center.x = state.at<float>(0);
        center.y = state.at<float>(1);
        if (plot)
        {
            plot_prediction(display_img);
        }
    }
}

void kalman_filt::plot_prediction(cv::Mat *display_img)
{
    cv::circle((*display_img), center, 2, cv::Scalar(255, 0, 0), -1);
    cv::rectangle((*display_img), predRect, cv::Scalar(255, 0, 0), 2);
}

void kalman_filt::plot_post(cv::Mat *display_img)
{
    cv::circle((*display_img), center, 2, cv::Scalar(0, 140, 255), -1);
    cv::rectangle((*display_img), postRect, cv::Scalar(0, 140, 255), 2);
}

void kalman_filt::get_time()
{
    tick_prev = tick;
    tick = std::chrono::system_clock::now();
    double DT = std::chrono::duration_cast<std::chrono::milliseconds>(tick - tick_prev).count() / 1000.0;
}

void kalman_filt::update(cv::Rect measure_box, bool state_track, cv::Mat *display_img, bool plot)
{
    if (state_track)
    {
        notFoundCount++;
        cout << "notFoundCount:" << notFoundCount << endl;
        if (notFoundCount >= 1000)
        {
            found = false;
        }
        else
            kf.statePost = state;
    }
    else
    {
        notFoundCount = 0;

        meas.at<float>(0) = measure_box.x + measure_box.width / 2;
        meas.at<float>(1) = measure_box.y + measure_box.height / 2;
        meas.at<float>(2) = (float)measure_box.width;
        meas.at<float>(3) = (float)measure_box.height;

        if (!found) // First detection!
        {
            // >>>> Initialization
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px

            state.at<float>(0) = meas.at<float>(0);
            state.at<float>(1) = meas.at<float>(1);
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;
            state.at<float>(4) = meas.at<float>(2);
            state.at<float>(5) = meas.at<float>(3);
            // <<<< Initialization

            kf.statePost = state;

            found = true;
        }
        else
            kf.correct(meas); // Kalman Correction

        // cout << "Measure matrix:" << endl
        //      << meas << endl;
    }
    postRect.width = kf.statePost.at<float>(4);
    postRect.height = kf.statePost.at<float>(5);
    postRect.x = kf.statePost.at<float>(0) - postRect.width / 2;
    postRect.y = kf.statePost.at<float>(1) - postRect.height / 2;

    postCenter.x = kf.statePost.at<float>(0);
    postCenter.y = kf.statePost.at<float>(1);
    if (plot)
        plot_post(display_img);
}