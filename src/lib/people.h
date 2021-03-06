#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <math.h>
#include "configurator.h"
#include "utils.h"
#include "Hungarian.h"
#include "nms.h"

namespace TrackerPeople
{

    class Person
    {
    private:
        /* data */
    public:
        Person(/* args */);
        ~Person();
        cv::Rect2i Box, Box_scaled;
        cv::Point2i centre;
        int ID = -1;
        bool is_user = false;
        int counter_lost = 0;
        void set_position(cv::Rect2i rect);
        // histogram descriptor
        std::vector<std::vector<float>> hist_descriptor = {std::vector<float>(50, 0), std::vector<float>(50, 0), std::vector<float>(50, 0)};

        std::vector<float> reid_descriptor_prev = std::vector<float>(256, 0);
        std::vector<float> reid_descriptor = std::vector<float>(256, 0);

        std::vector<cv::Point2i> story;
    };

    class PeopleList
    {
    private:
        /* data */
        cv::Mat img, img_HSV;

        bool first_loop = true;
        int max_counter_lost = 60;

        int max_id = 0;
        std::string detector_xml, detector_bin;
        std::string reid_xml, reid_bin;
        cv::dnn::Net dnn_detector, dnn_reidentificator;

        std::vector<cv::Rect2i> detection_prev;
        std::vector<cv::Rect2i> detection;

        std::vector<std::vector<float>> descriptors_prev;
        std::vector<std::vector<float>> descriptors;

        std::vector<std::vector<double>> costMatrix;
        HungarianAlgorithm HungAlgo;
        std::vector<int> HungAssignment;

    public:
        std::vector<Person> list;
        std::vector<Person> list_prev;
        std::vector<Person> list_new;

        int index_user_in_list;
        TrackerPeople::Person user;

        void update(cv::Mat img);
        void detectPeoples();
        void extractPeopleDescriptors();

        void populateList();

        void generateCostMatrix();
        void doHungarian();

        void sortNewDetection();
        void extractUser();
        void checkToKill();

        void plotBoxes(cv::Mat &img);
        void plotTrack(cv::Mat &img);
        void plotStoryPosition(cv::Mat &img);

        PeopleList();
        ~PeopleList();
    };
}