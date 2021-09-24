#include "pose_gesture.h"

void pose_keypoints::get_pose(std::vector<int> personKeypoints,
                              std::vector<KeyPoint_pose> keyPointsList)
{
    /* code */
    pose.clear();

    for (int i = 0; i < personKeypoints.size(); i++)
    {

        int index = personKeypoints[i];
        if (index != -1)
        {
            KeyPoint_pose &kp_i = keyPointsList[index];
            pose.push_back(kp_i);
        }
        else
        {
            KeyPoint_pose kp_i = KeyPoint_pose(cv::Point(-1, -1), -1);
            pose.push_back(kp_i);
        }
    }

    bool flag_neck = get_kp(NOSE).probability == -1;
    bool flag_shoulder = get_kp(RIGHT_SHOULDER).probability == -1 | get_kp(LEFT_SHOULDER).probability == -1;
    bool flag_elbow = get_kp(RIGHT_ELBOW).probability == -1 | get_kp(LEFT_ELBOW).probability == -1;
    bool flag_wrist = get_kp(RIGHT_WRIST).probability == -1 | get_kp(LEFT_WRIST).probability == -1;

    have_keypoints = flag_neck | flag_shoulder | flag_elbow | flag_wrist;
};

KeyPoint_pose pose_keypoints::get_kp(int idx)
{
    return pose[idx];
};

void pose_keypoints::get_gesture()
{
    if (!have_keypoints)
    {
        // std::cout << "in Gesture recognition";
        KeyPoint_pose nose = get_kp(NOSE);
        KeyPoint_pose neck = get_kp(NECK);
        KeyPoint_pose r_wrist = get_kp(RIGHT_WRIST);
        KeyPoint_pose l_wrist = get_kp(LEFT_WRIST);
        KeyPoint_pose r_elbow = get_kp(RIGHT_ELBOW);
        KeyPoint_pose l_elbow = get_kp(LEFT_ELBOW);
        KeyPoint_pose r_shoulder = get_kp(RIGHT_SHOULDER);
        KeyPoint_pose l_shoulder = get_kp(LEFT_SHOULDER);

        // evaluate the shoulder width
        float shoulder_width = distance(l_shoulder.point, r_shoulder.point);
        float hands_distance = distance(l_wrist.point, r_wrist.point);
        float vertical_angle_right_arm = vertical_angle(r_wrist.point, r_elbow.point);
        float vertical_angle_left_arm = vertical_angle(l_wrist.point, l_elbow.point);

        bool left_hand_up = l_wrist.point.y < neck.point.y;
        bool right_hand_up = r_wrist.point.y < neck.point.y;
        bool large_distance_hands = hands_distance > shoulder_width;

        float near_dist = shoulder_width / (float)1.5;
        bool distance_neck = distance(r_wrist.point, neck.point) < near_dist & distance(l_wrist.point, neck.point) < near_dist;
        float angle_left = angle(l_shoulder.point, l_elbow.point, l_wrist.point);
        float angle_righ = angle(r_wrist.point, r_elbow.point ,r_shoulder.point);
        bool angle_elbows = angle_righ <= 70.0 & angle_left <= 70.0;
        bool right_hand_on_neck = r_wrist.point.x < r_shoulder.point.x & r_wrist.point.y < r_shoulder.point.y;
        bool left_hand_on_neck = l_wrist.point.x > l_shoulder.point.x & l_wrist.point.y < l_shoulder.point.y;

        // cout << " distances bool  " << distance_neck << endl;
        // cout << " angle bool  r: " << (angle_righ <= 70.0) << "  l: " << (angle_left <= 70.0) << endl;
        // cout << "\nangles  r: " <<  angle_righ << " -- l " << angle_left << "\n";

        if (left_hand_up & right_hand_up) // & large_distance_hands) // control HANDS_UP
        {
            gm.actual_gesture = HANDS_UP;
            // std::cout << "   HANDS_UP" << endl;
        }
        else if (distance_neck & !(left_hand_up | right_hand_up) & angle_elbows) // control HANDS_ON_NECK
        {
            gm.actual_gesture = HANDS_ON_NECK;
            // std::cout << "   HANDS_ON_NECK" << endl;
        }
        else
        {
            gm.actual_gesture = RESET;
            // std::cout << "   RESET" << endl;
        }
    }
    else
    {
        gm.actual_gesture = RESET;
        // std::cout << "   RESET" << endl;
    }
}

void pose_keypoints::gesture_to_action()
{

    // exist already active gesture (?) -> check the gesture manager
    if (!have_keypoints)
    {
        int prev_gesture_capt = gm.actual_gesture;
        get_gesture();

        switch (gm.robot_action)
        {
        case IDLE:
            /* wait for gesture */
            if (gm.gesture == HANDS_ON_NECK)
            {
                // move to follow -> iff gesture for more than min_time \2000 ms
                // init timer
                if (gm.flag_capture == false)
                {
                    gm.flag_capture = true;
                    gm.timer_start = std::chrono::system_clock::now();
                    gm.timer_tmp = gm.timer_start;
                }

                if (gm.actual_gesture == prev_gesture_capt)
                {
                    gm.timer_tmp_prev = gm.timer_tmp;
                    gm.timer_tmp = std::chrono::system_clock::now();
                    gm.time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(gm.timer_tmp - gm.timer_tmp_prev).count();
                }

                // if
                if (std::chrono::duration_cast<std::chrono::milliseconds>(gm.timer_start - gm.timer_tmp).count() > gm.max_time)
                {
                    gm.flag_capture = false;
                }

                // gesture done for enought time -> go to FOLLOW
                if (gm.time_ms > gm.min_time)
                    gm.robot_action = FOLLOW;
            }
            else if (gm.gesture == HANDS_UP) // if detect stop action, immediatly stop if prev action and actual are the same
            {
                if (prev_gesture_capt == HANDS_UP)
                    gm.robot_action = STOP;
                else
                    gm.robot_action = IDLE;
            }
            else
            {
                // do nothing, stay idle
                gm.robot_action = IDLE;
            }

            break; // <<<<< IDLE

        case STOP:
            /* code */
            if (gm.flag_capture == false)
            {
                gm.flag_capture = true;
                gm.timer_start = std::chrono::system_clock::now();
                gm.timer_tmp = gm.timer_start;
            }
            else
            {
                gm.timer_tmp_prev = gm.timer_tmp;
                gm.timer_tmp = std::chrono::system_clock::now();
                gm.time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(gm.timer_tmp - gm.timer_tmp_prev).count();
            }
            if (gm.time_ms > 30000)
                gm.robot_action = IDLE;

            break; // <<<<< STOP

        case FOLLOW:
            if (gm.gesture == HANDS_UP) // if detect stop action, immediatly stop if prev action and actual are the same (avoid errors)
            {
                if (prev_gesture_capt == HANDS_UP)
                    gm.robot_action = STOP;
                else
                    gm.robot_action = FOLLOW;
            }
            break; // <<<<< FOLLOW
        }
    }
    else
    {
        // se supero tot tempo senza identificare una gesture gm.gesture = RESET;
    }
};
