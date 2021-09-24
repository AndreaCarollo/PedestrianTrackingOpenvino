//                                  _                   _
//  _ __  ___  ___ ___    ___ __ __| |_  _ _  __ _  __ | |_  ___  _ _
// | '_ \/ _ \(_-</ -_)  / -_)\ \ /|  _|| '_|/ _` |/ _||  _|/ _ \| '_|
// | .__/\___//__/\___|  \___|/_\_\ \__||_|  \__,_|\__| \__|\___/|_|
// |_|
//

#include "pose_extractor.h"
std::vector<std::vector<int>> get_poses(cv::Mat *img, cv::dnn::Net *network_pose,
                                        std::vector<KeyPoint_pose> *keyPointsList,
                                        bool filter_poses, int min_keypoint, float scale_blob, float thresh)
{
    // get image size
    cv::Size img_size = img->size();

    // generate the blob from the image
    cv::Mat inputBlob = cv::dnn::blobFromImage((*img), 1.0f, Size((int)img_size.width / scale_blob, (int)img_size.height / scale_blob), Scalar(), false, false);

    // set up the input blob
    network_pose->setInput(inputBlob);

    // do convolution for the layer requested \outBlobNames
    std::vector<cv::Mat> output_net;
    network_pose->forward(output_net, outBlobNames);

    //extract heatmap and part affinity map from the blob vector
    cv::Mat heatmap_blob = output_net[0];
    cv::Mat paf_blob = output_net[1];

    // std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();

    // extract all points from the heat Map blob
    vector<vector<KeyPoint_pose>> detectedKeypoints;
    extract_all_keypoints(heatmap_blob, img_size, &detectedKeypoints, keyPointsList, thresh);

    // get valid pairs of keypoints in according to Part Affinity blob
    std::vector<std::vector<ValidPair>> validPairs;
    std::set<int> invalidPairs;
    std::vector<cv::Mat> pafs_tot = getValidPairs(paf_blob, img_size, detectedKeypoints, &validPairs, &invalidPairs);

    // std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();
    // std::cout << "Time Taken in pose estimation PEAK = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms " << std::endl;
   
    // connect all valid pairs to generate skeletons
    std::vector<std::vector<int>> personwiseKeypoints;
    getPersonwiseKeypoints(validPairs, invalidPairs, personwiseKeypoints);


    if (filter_poses)
    {
        // keep only skeleton with at least \min_keypoint = 6 keypoints
        std::vector<std::vector<int>> good_skeleton = good_skeleton_extraction(personwiseKeypoints, min_keypoint);
        return good_skeleton;
    }
    else
    {
        return personwiseKeypoints;
    }
}

std::vector<std::vector<int>> get_poses_cov(cv::Mat *img, cv::dnn::Net *network_pose,
                                            std::vector<KeyPoint_pose> *keyPointsList,
                                            bool filter_poses, int min_keypoint, float scale_blob, float thresh)
{
    // get image size
    cv::Size img_size = img->size();

    // generate the blob from the image
    cv::Mat inputBlob = cv::dnn::blobFromImage((*img), 1.0f, Size((int)img_size.width / scale_blob, (int)img_size.height / scale_blob), Scalar(), false, false);

    // set up the input blob
    network_pose->setInput(inputBlob);

    // do convolution for the layer requested \outBlobNames
    std::vector<cv::Mat> output_net;
    network_pose->forward(output_net, outBlobNames);

    //extract heatmap and part affinity map from the blob vector
    cv::Mat heatmap_blob = output_net[0];
    cv::Mat paf_blob = output_net[1];
    std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();

    // extract all points from the heat Map blob
    vector<vector<KeyPoint_pose>> detectedKeypoints;
    extract_all_keypoints_cov(heatmap_blob, img_size, &detectedKeypoints, keyPointsList, img, false, false, thresh);

    // get valid pairs of keypoints in according to Part Affinity blob
    std::vector<std::vector<ValidPair>> validPairs;
    std::set<int> invalidPairs;
    std::vector<cv::Mat> pafs_tot = getValidPairs(paf_blob, img_size, detectedKeypoints, &validPairs, &invalidPairs);

    std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();
    std::cout << "Time Taken in pose estimation COV = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms " << std::endl;
    
    // connect all valid pairs to generate skeletons
    std::vector<std::vector<int>> personwiseKeypoints;
    getPersonwiseKeypoints(validPairs, invalidPairs, personwiseKeypoints);


    if (filter_poses)
    {
        // keep only skeleton with at least \min_keypoint = 6 keypoints
        std::vector<std::vector<int>> good_skeleton = good_skeleton_extraction(personwiseKeypoints, min_keypoint);
        return good_skeleton;
    }
    else
    {
        return personwiseKeypoints;
    }
}

std::vector<int> parse_poses(cv::Rect *ROI, std::vector<KeyPoint_pose> keyPointsList,
                             std::vector<std::vector<int>> *poses, cv::Mat *img)
{
    int min_counter = 4;
    int index_pose = -1;
    float min_area_score = 0.45;

    cv::Rect out_box;

    std::vector<int> return_pose;
    std::vector<KeyPoint_pose> point_list;

    for (int n = 0; n < poses->size(); n++)
    {
        int counter_critic_point = 0;
        int counter = 0;
        std::vector<cv::Point> point_list;
        std::vector<int> list_x;
        std::vector<int> list_y;
        for (int i = 0; i < 17; i++)
        {
            int index = poses->at(n)[i];
            if (index == -1)
            {
                continue;
            }
            KeyPoint_pose &kp = keyPointsList[index];
            list_x.push_back(kp.point.x);
            list_y.push_back(kp.point.y);

            // check if contain the keypoint
            if (ROI->contains(cv::Point((int)kp.point.x, (int)kp.point.y)))
            {
                // increase counter for keypoint
                counter++;
                // cout << "contiene kp " << i << endl;

                // consider if the ROI contains some critic points
                if (i == 1 | i == 5 | i == 2 | i == 0)
                {
                    counter_critic_point++;
                }
            }
        }
        // build rectangle containing the pose
        int x_min = *min_element(list_x.begin(), list_x.end());
        int y_min = *min_element(list_y.begin(), list_y.end());
        int x_max = *max_element(list_x.begin(), list_x.end());
        int y_max = *max_element(list_y.begin(), list_y.end());
        cv::Rect box_pose = Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max));

        // evaluate a score: rate of intersecated rect over area of ROI
        float score = (float)(box_pose & (*ROI)).area() / (float)box_pose.area();
        bool tmp = (box_pose | (*ROI)).area() < (box_pose.area() + (*ROI).area());

        if (score > min_area_score & tmp) //& counter_critic_point >= 2)
        {
            if (counter_critic_point > 3)
            {
                index_pose = n;
                out_box = box_pose;
            }
            else
            {
                min_area_score = score;
                // min_counter = counter;
                index_pose = n;
                out_box = box_pose;
            }
        }
    }
    // cout << " >> pose index " << index_pose;
    if (index_pose == -1)
    {
        return nullpose;
    }
    else
    {
        (*ROI) = (out_box | (*ROI));
        return_pose = poses->at(index_pose);
        (*poses)[index_pose] = nullpose;
        return return_pose;
    }
};