#include "dnn_loader.h"

using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

cv::dnn::Net load_detection_people()
{
    std::string model_PED = "/home/andrea/openvino_models/ir/intel/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.xml";
    std::string config_PED = "/home/andrea/openvino_models/ir/intel/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.bin";
    cv::dnn::Net network_Pedestrian = readNetFromModelOptimizer(model_PED, config_PED);
    network_Pedestrian.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    network_Pedestrian.setPreferableTarget(DNN_TARGET_CPU);
    return network_Pedestrian;
}

cv::dnn::Net load_detection_faces()
{
    std::string model_FACE = "/home/andrea/openvino_models/ir/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml";
    std::string config_FACE = "/home/andrea/openvino_models/ir/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.bin";
    cv::dnn::Net network_faces = readNetFromModelOptimizer(model_FACE, config_FACE);
    network_faces.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    network_faces.setPreferableTarget(DNN_TARGET_CPU);
    return network_faces;
};

cv::dnn::Net load_reid_people()
{
    // std::string model_Feature_PED = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.xml";
    // std::string config_Feature_PED = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.bin";

    // works, but not enought accurate (~77%)
    std::string model_Feature_PED = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0031/FP32/person-reidentification-retail-0031.xml";
    std::string config_Feature_PED = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0031/FP32/person-reidentification-retail-0031.bin";
    cv::dnn::Net network_Feature_Body = readNet(model_Feature_PED, config_Feature_PED);
    network_Feature_Body.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    network_Feature_Body.setPreferableTarget(DNN_TARGET_CPU);
    return network_Feature_Body;
};

cv::dnn::Net load_reid_faces()
{
    std::string model_Feature_Face = "/home/andrea/openvino_models/ir/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml";
    std::string config_Feature_Face = "/home/andrea/openvino_models/ir/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin";
    cv::dnn::Net net_face_feature = cv::dnn::readNetFromModelOptimizer(model_Feature_Face, config_Feature_Face);
    net_face_feature.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    net_face_feature.setPreferableTarget(DNN_TARGET_CPU);
    return net_face_feature;
};

void load_net_pose(cv::dnn::Net *network_pose)
{

    std::string model_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml";
    std::string config_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin";

    //* load neural network and config
    (*network_pose) = readNetFromModelOptimizer(model_POSE, config_POSE);

    //* Set backend and target to execute network
    network_pose->setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    network_pose->setPreferableTarget(DNN_TARGET_CPU);
}
