#ifndef POSE_GESTURE
#define POSE_GESTURE

//   ___           _                          _              _
//  / __| ___  ___| |_  _  _  _ _  ___    __ | |_   ___  __ | |__
// | (_ |/ -_)(_-<|  _|| || || '_|/ -_)  / _|| ' \ / -_)/ _|| / /
//  \___|\___|/__/ \__| \_,_||_|  \___|  \__||_||_|\___|\__||_\_\
//

#include "pose_basic.h"
#include "pose_utils.h"

// struct pose_keypoint
// {
//     /* data */
//     KeyPoint_pose nose;
//     KeyPoint_pose neck;
//     KeyPoint_pose r_wrist;
//     KeyPoint_pose l_wrist;
//     KeyPoint_pose r_elbow;
//     KeyPoint_pose l_elbow;
//     KeyPoint_pose r_shoulder;
//     KeyPoint_pose l_shoulder;

// };
enum active_gesture
{
    RESET = 0,
    HANDS_UP = 1,
    HANDS_ON_NECK = 2
};

enum robot_action
{
    IDLE = 0,
    STOP = 1,
    FOLLOW = 2
};

struct gesture_manager
{
    /* data */
    int gesture = RESET;
    int actual_gesture = RESET;
    int robot_action = IDLE;
    std::chrono::time_point<std::chrono::system_clock> timer_start;
    std::chrono::time_point<std::chrono::system_clock> timer_tmp;
    std::chrono::time_point<std::chrono::system_clock> timer_tmp_prev;

    int time_ms = 0;
    int max_time = 5000; // ms
    int min_time = 2000; // ms

    bool flag_capture = false;
    bool flag_reset_timer = false;

};

class pose_keypoints
{
private:
    /* data */
public:
    vector<KeyPoint_pose> pose;
    gesture_manager gm;
    bool have_keypoints = false;

    KeyPoint_pose get_kp(int idx);
    void get_pose(std::vector<int> personKeypoints, std::vector<KeyPoint_pose> keyPointsList);
    void get_gesture();
    void gesture_to_action();
};

#endif