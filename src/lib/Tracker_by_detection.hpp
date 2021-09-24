#include <opencv2/dnn.hpp>
#include "utils.h"
#include "kalman_filt.h"


#ifndef TRACKER_by_DET
#define TRACKER_by_DET

class Tracker_by_detection
{
private:
    /* data Detection Net */// pedestrian-detection-adas-0002
    float Confidence_Ped = 0.90;
    float blobscale_Pedestrian = 1.0;
    cv::Size sizeBlob_Pedestrian = cv::Size(672, 384);
    float scaleOut_Pedestrian = 1.0 / 255.0;
    cv::Scalar meanOut_Pedestrian = 0;
    float minimo_feature_match = 0.7f;

    /* data Feature Net */// person-reidentification-retail-0031
    cv::Size sizeBlobFeature_Pedestrian = cv::Size(256, 128);

public:

    // Nets
    cv::dnn::Net *net_reid;
    cv::dnn::Net *net_detection;
    cv::dnn::Net *net_feat_face;
    int imgCols;
    int imgRows;

    // Kalman filter for position estimation
    kalman_filt kf;

    // UWB data
    float UWB_angle = 0.0; // degree
    float UWB_distance = 0.0; // meter
    float UWB_x_position = 0.0; // meter

    float RS2_angle = 0.0; // degree
    float RS2_distance = 0.0; // meter
    float RS2_x_position = 0.0; // meter

    // Data of the Operator
    cv::Rect box;
    std::vector<float> initial_features;
    std::vector<float> actual_features;
    std::vector<float> face_features;
    cv::Mat initCutFrame;

    std::vector<cv::Point> Trace_2D;

    /* Candidates data */
    // RECT people in the frame
    std::vector<cv::Rect> people_in_frame;
    std::vector<std::vector<float>> candidate_features;

    // Flags
    bool featureAreUpdated = false;
    bool lost_flag = true;
    int lost_counter = 0;

    // Functions
    void set_up_net(cv::dnn::Net *net_reid_pointer, cv::dnn::Net *net_detection_pointer, cv::dnn::Net *net_reid_face_pointer);
    void saveImgSize(cv::Mat img);
    
    std::vector<float> extract_vector_feature(cv::Mat img);
    void set_up_BOX(cv::Rect ROI);
    void set_up_feature(cv::Mat img);
    void reset_flag_in_detection();
    void save_face_feature( std::vector<float> face_feature_init);
    bool update_feature_and_track(cv::Mat img);
    void detect_people_on_frame(cv::Mat img);
    void extract_candidate_feature(cv::Mat img);
    void updateTrace2D();
    void update_trace3D();
    void plot_PositionHistory(cv::Mat *img);
};

void condition_detection(cv::Mat img,std::vector<cv::Rect> bodies , std::vector<cv::Rect> faces, Tracker_by_detection *User, parameters *param);


#endif


