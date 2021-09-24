#include "Tracking.h"

void do_tracking(cv::Mat *img, cv::Mat *display_img,
                 Tracker_by_detection *UserToTrack, parameters *param,
                 cv::dnn::Net *network_Pedestrian, cv::dnn::Net *network_faces,
                 cv::dnn::Net *network_Feature_Body)
{
    // >>>> Kalman prediction
    UserToTrack->kf.get_time();
    UserToTrack->kf.prediction(display_img, false);

    bool update_track = false;

    // std::vector<cv::Rect> DETECTION_people_in_frame, DETECTION_faces_in_frame, REID_people_in_frame;
    std::vector<float> REID_people_feature_in_frame;

    // >>>> Tracking
    switch (param->current_state)
    {
    case DETECT:
        std::cout << " -->> DETECTION ---  ";
        //_____ Detection __________________________________________________________________________
        param->DETECTION_people_in_frame = DetectionPedestrian_on_frame((*img), network_Pedestrian);

        param->DETECTION_faces_in_frame = DetectionFaces_on_frame((*img), network_faces);

        //_____ Condition change state _____________________________________________________________
        condition_detection((*img), param->DETECTION_people_in_frame, param->DETECTION_faces_in_frame, UserToTrack, param);
        UserToTrack->reset_flag_in_detection();

        //_____ Plot Detection _____________________________________________________________________
        for (int i = 0; i < param->DETECTION_faces_in_frame.size(); i++)
        {
            cv::rectangle((*display_img), param->DETECTION_faces_in_frame[i], cv::Scalar(0, 255, 255), 2, 8, 0);
        }
        // plot_In_Out_objects(&display_img, &param);
        for (int i = 0; i < param->DETECTION_people_in_frame.size(); i++)
        {
            /* code */
            cv::rectangle((*display_img), param->DETECTION_people_in_frame[i], cv::Scalar(0, 255, 255), 2, 8, 0);
        }
        break; /* <<< DETECTION */

    case TRACK:
        std::cout << " -->> TRACKING ---  ";
        //_____ DO Tracking ______________________________________________________________________
        // Detect people in frame
        UserToTrack->detect_people_on_frame((*img));
        // Extract feature of people in frame
        UserToTrack->extract_candidate_feature((*img));
        // Update the traker
        update_track = UserToTrack->update_feature_and_track((*img));
        // Update kalman filter
        UserToTrack->kf.update(UserToTrack->box, param->current_state != TRACK, display_img, false);
        // Update the 2D trace on frame
        UserToTrack->updateTrace2D();
        UserToTrack->plot_PositionHistory(display_img);

        if (UserToTrack->lost_flag)
        {
            param->current_state = REIDENTIFICATION;
        }

        // for (int i = 0; i < param->DETECTION_people_in_frame.size(); i++)
        // {
        //     /* code */
        //     cv::rectangle((*display_img), UserToTrack->people_in_frame[i], cv::Scalar(0, 255, 255), 2, 8, 0);
        // }

        cv::rectangle((*display_img), UserToTrack->box, cv::Scalar(0, 255, 0), 2, 8, 0);
        break; /* <<< TRACKING */

    case REIDENTIFICATION:
        std::cout << "reidentification to do" << std::endl;
        //_____ Detection __________________________________________________________________________
        param->REID_people_in_frame = DetectionPedestrian_on_frame((*img), network_Pedestrian);
        /*  TODO:
             - estrarre feature dei corpi
             - rematch con il corpo iniziale
             - if dopo tot cicli/tempo non lo trovo -> uso la faccia
             - reinitialize the tracker by detection
             - use information of the kalman prediction to get new position
            */
        for (int i = 0; i < param->REID_people_in_frame.size(); i++)
        {
            rectangle((*display_img), param->REID_people_in_frame[i], cv::Scalar(0, 0, 255));
        }
        // plots:
        // // DetectionFaces_on_frame(img, network_faces, &param, false);
        // // for (int i = 0; i < param.oggetti_Face.size(); i++)
        // // {
        // //     cv::rectangle(display_img, param.oggetti_Face[i], cv::Scalar(0, 255, 255), 2, 8, 0);
        // // }
        // // for (int i = 0; i < param.oggetti_People.size(); i++)
        // // {
        // //     cv::rectangle(display_img, param.oggetti_People[i], cv::Scalar(0, 255, 0), 2, 8, 0);
        // // }

        resizeRects(&param->REID_people_in_frame, UserToTrack->imgCols, UserToTrack->imgRows);
        for (int i = 0; i < param->REID_people_in_frame.size(); i++)
        {
            REID_people_feature_in_frame = extract_body_feature(network_Feature_Body, (*img)(param->REID_people_in_frame[i]));
            // std::cout << "cos sim:" << cosine_similarity(REID_people_feature_in_frame, UserToTrack->initial_features) << std::endl;
            if (cosine_similarity(REID_people_feature_in_frame, UserToTrack->initial_features) > 0.56)
            {
                // UserToTrack.box = REID_people_in_frame[i];
                UserToTrack->set_up_BOX(param->REID_people_in_frame[i]); // set up box of the user
                rectangle((*display_img), UserToTrack->box, cv::Scalar(0, 255, 0));
                UserToTrack->reset_flag_in_detection();

                param->current_state = TRACK;
                UserToTrack->kf.init();
                i = param->REID_people_in_frame.size() + 10;
            }
            //else{
            // TODO: dopo tot tempo che rimango in reid -> stop the robot, torna a detection con l'identificazione facciale
            // }
        }
        break; /* <<< REIDENTIFICATION */
    };

    // >>>>> Kalman Update
    // if (update_track)
    //     UserToTrack->kf.update(UserToTrack->box, param->current_state != TRACK, display_img, false);

    // cv::rectangle((*display_img), UserToTrack->box, cv::Scalar(0, 255, 0), 2, 8, 0);
}