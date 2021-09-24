#include "Tracker_by_detection.hpp"

void Tracker_by_detection::set_up_net(cv::dnn::Net *net_reid_pointer, cv::dnn::Net *net_detection_pointer, cv::dnn::Net *net_reid_face_pointer)
{
    net_reid = net_reid_pointer;
    net_detection = net_detection_pointer;
    net_feat_face = net_reid_face_pointer;

    // initialize kalman filter
    kf.init();
}

void Tracker_by_detection::saveImgSize(cv::Mat img)
{
    // save img dimension
    imgCols = img.cols;
    imgRows = img.rows;
}
void Tracker_by_detection::set_up_BOX(cv::Rect ROI)
{
    box = ROI;
}

std::vector<float> Tracker_by_detection::extract_vector_feature(cv::Mat img)
{
    std::vector<float> feature = extract_body_feature(net_reid, img);
    return feature;
}

void Tracker_by_detection::set_up_feature(cv::Mat img)
{
    initial_features = Tracker_by_detection::extract_vector_feature(img);
    actual_features = initial_features;
}
void Tracker_by_detection::reset_flag_in_detection()
{
    lost_flag = false;
    lost_counter = 0;
}

void Tracker_by_detection::save_face_feature(std::vector<float> face_feature_init)
{

    face_features = face_feature_init;
}

bool Tracker_by_detection::update_feature_and_track(cv::Mat img)
{
    featureAreUpdated = false;
    float tmp_cosine_similarity;
    double cosine_minimo = 0.7; // minimo_feature_match;
    double score_intersection_minimo = 0.55;
    std::vector<float> tmp_new_feature;
    cv::Rect tmp_new_box;
    bool flag_find = false;

    // bool overlap_kalman = false;

    // for (int i = 0; i < people_in_frame.size(); i++)
    // {
    //     if ((kf.predRect & people_in_frame[i]).area() > 0.0)
    //     {
    //         overlap_kalman = true;
    //         break;
    //     }
    // }

    // if (overlap_kalman)
    // {

    // given candidate_features have to find max cosine
    for (int i = 0; i < candidate_features.size(); i++)
    {
        tmp_cosine_similarity = cosine_similarity(candidate_features[i], actual_features);
        double score_intersection = scoreOverlap(people_in_frame[i], kf.predRect);
        // std::cout << "intersection score " << score_intersection;
        // std::cout << " - cosine score " << tmp_cosine_similarity << std::endl;
        if ((tmp_cosine_similarity > cosine_minimo) & score_intersection > score_intersection_minimo) // & tmp_cosine_similarity < 0.90 //& score_intersection > 0.4)
        {
            cosine_minimo = tmp_cosine_similarity;
            score_intersection_minimo = score_intersection;
            tmp_new_feature = candidate_features[i];
            tmp_new_box = people_in_frame[i];
            flag_find = true;
        }
    }
    // }
    // change feature_are_updated + update BOX + update TRACK
    // TODO: add information of kalman filter
    if (flag_find)
    {
        actual_features = tmp_new_feature;
        box = tmp_new_box;
    }
    // else
    // {
    //     // actual_features = initial_features;
    //     box = kf.predRect;
    //     lost_counter++;
    //     if (lost_counter == 30)
    //     {
    //         lost_flag = true;
    //     }
    // }

    featureAreUpdated = flag_find;
    return flag_find;
}

void Tracker_by_detection::detect_people_on_frame(cv::Mat img)
{
    people_in_frame = DetectionPedestrian_on_frame(img, net_detection);
    resizeRects(&people_in_frame, imgCols, imgRows);
}

void Tracker_by_detection::extract_candidate_feature(cv::Mat img)
{
    std::vector<std::vector<float>> tmp_candidate_features;
    std::vector<float> tmp_single_feature;
    cv::Mat crop_img;
    for (int i = 0; i < people_in_frame.size(); i++)
    {
        /* extract feature for all detections */
        cv::Rect resized_box = people_in_frame[i];
        resizeRect(&resized_box, imgCols, imgRows);
        img(resized_box).copyTo(crop_img);
        tmp_single_feature = extract_vector_feature(crop_img);
        tmp_candidate_features.push_back(tmp_single_feature);
    }
    candidate_features = tmp_candidate_features;
}

// Detection condition
void condition_detection(cv::Mat img, std::vector<cv::Rect> bodies, std::vector<cv::Rect> faces, Tracker_by_detection *User, parameters *param)
{
    // identification of the right face
    std::vector<float> feature_to_control;
    // personToTrack->gotFace = false;
    cv::Rect the_face;
    cv::Rect the_body;
    bool got_face = false;
    float Max_cosine = 0.5;
    resizeRects(&faces, img.cols, img.rows);

    for (int i = 0; i < faces.size(); i++)
    {
        /* code */
        feature_to_control = extract_face_feature((*User->net_feat_face), img(faces[i]));
        if (cosine_similarity(feature_to_control, User->face_features) > Max_cosine)
        {
            the_face = faces[i];
            got_face = true;
        }
    }

    resizeRects(&bodies, img.cols, img.rows);
    if (findHisBody(bodies, the_face, &the_body))
    {

        User->set_up_BOX(the_body); // set up box of the user
        cv::Mat tmp;
        img(the_body).copyTo(tmp);
        User->set_up_feature(tmp); // set initial & actual features;
        img(the_body).copyTo(User->initCutFrame);

        std::cout << "change in track" << std::endl;
        param->current_state = TRACK;
    }
    else
    {
        param->current_state = DETECT;
    }
}

void Tracker_by_detection::updateTrace2D()
{
    cv::Point centroid = getCentre_rect(box);

    Trace_2D.push_back(cv::Point(centroid.x, centroid.y));
    if (Trace_2D.size() > 30)
    {
        Trace_2D.erase(Trace_2D.begin(), Trace_2D.begin() + (Trace_2D.size() - 30));
    }
}

void Tracker_by_detection::plot_PositionHistory(cv::Mat *img)
{
    for (int i = 0; i < Trace_2D.size() - 1; i++)
    {
        cv::Point2i pt1 = cv::Point2i(Trace_2D[i].x, Trace_2D[i].y);
        cv::Point2i pt2 = cv::Point2i(Trace_2D[i + 1].x, Trace_2D[i + 1].y);
        cv::line((*img), pt1, pt2, cv::Scalar(0, 0, 255), 1.5, 8);
    }
}