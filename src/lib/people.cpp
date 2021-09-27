#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include "people.h"

namespace TrackerPeople
{

    Person::Person(/* args */)
    {
    }

    Person::~Person()
    {
    }

    void Person::set_position(cv::Rect2i rect)
    {
        Box = rect;
    }

    PeopleList::PeopleList()
    {
        this->reid_xml = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0031/FP16-INT8/person-reidentification-retail-0031.xml";
        this->reid_bin = "/home/andrea/openvino_models/ir/intel/person-reidentification-retail-0031/FP16-INT8/person-reidentification-retail-0031.bin";

        this->detector_xml = "/home/andrea/openvino_models/ir/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml";
        this->detector_bin = "/home/andrea/openvino_models/ir/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.bin";

        this->dnn_detector = cv::dnn::readNetFromModelOptimizer(detector_xml, detector_bin);
        this->dnn_detector.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        this->dnn_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        this->dnn_reidentificator = cv::dnn::readNetFromModelOptimizer(reid_xml, reid_bin);
        this->dnn_reidentificator.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        this->dnn_reidentificator.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    PeopleList::~PeopleList()
    {
    }

    void PeopleList::update(cv::Mat img)
    {
        this->img = img;
        cv::cvtColor(img, img_HSV, cv::COLOR_BGR2HSV);
        this->checkToKill();
        this->detectPeoples();
        this->extractPeopleDescriptors();
        this->populateList();
        if (first_loop)
        {
            first_loop = false;
            list = list_new;
            for (int i = 0; i < list.size(); i++)
            {
                /* code */
                this->max_id++;
                list[i].ID = max_id;
            }
        }
        else
        {
            this->generateCostMatrix();
            this->doHungarian();
            this->sortNewDetection();
            this->extractUser();
        }
    }

    void PeopleList::detectPeoples()
    {
        //cv::resize(img, img_1080, cv::Size(1920, 1080));
        this->detection_prev = this->detection;
        this->detection.clear();

        // set confidence level
        float Confidence_Ped = 0.6;

        // prepare blob
        float blobscale_Pedestrian = 1.0;                                    // / 16.0;
        cv::Size sizeBlob_Pedestrian = cv::Size(img.cols / 2, img.rows / 2); //cv::Size(320, 544);   //cv::Size(384, 672); //
        float scaleOut_Pedestrian = 1.0 / 16.0;                              // 16.0;
        1.0 / 255.0;
        cv::Scalar meanOut_Pedestrian = 1.0;
        // create blob from image
        cv::Mat blobFromImg_People;
        cv::dnn::blobFromImage(img, blobFromImg_People, blobscale_Pedestrian, sizeBlob_Pedestrian, false);

        //* Set blob as input of neural network.
        dnn_detector.setInput(blobFromImg_People, "", scaleOut_Pedestrian, meanOut_Pedestrian);

        // compure network
        cv::Mat outMat_People = dnn_detector.forward();
        // extract object from result of network
        auto data = (float *)outMat_People.data;

        // extract data
        std::vector<std::vector<float>> detection_vec;
        std::vector<cv::Rect> detection_rec;
        for (size_t i = 0; i < outMat_People.total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > Confidence_Ped)
            {
                std::cout << i << " confidence lvl  " << confidence << std::endl;
                int left = (int)(data[i + 3] * img.cols);
                int top = (int)(data[i + 4] * img.rows);
                int right = (int)(data[i + 5] * img.cols);
                int bottom = (int)(data[i + 6] * img.rows);

                // std::cout << i << " rectangle \t" << cv::Point2i(left, top) << cv::Point2i(right, bottom) << std::endl;

                detection_vec.push_back({(float)left, (float)top, (float)right, (float)bottom});
                detection_rec.push_back(cv::Rect(cv::Point2i(left, top), cv::Point2i(right, bottom)));
            }
        }

        detection = nms(detection_vec, 0.3);

        // todo NMS

        // resize rects
        resizeRects(&detection, img.cols, img.rows);
    }

    void PeopleList::extractPeopleDescriptors()
    {

        if (!first_loop)
            this->descriptors_prev = this->descriptors;
        this->descriptors.clear();
        for (int i = 0; i < this->detection.size(); i++)
        {
            // std::cout << "rect " << i << " \t  TL \t" << detection[i].tl() << " \t BR \t" << detection[i].br() << std::endl;
            auto tmp_ = extract_body_feature(&this->dnn_reidentificator, img(this->detection[i]));
            descriptors.push_back(tmp_);
        }
    }

    void PeopleList::populateList()
    {
        list_prev = list;
        list_new.clear();

        vector<cv::Mat> channels_HSV;
        // cv::split(img_HSV, channels_HSV);
        cv::split(img_HSV, channels_HSV);


        for (int i = 0; i < detection.size(); i++)
        {
            /* code */
            Person tmp_person;
            tmp_person.Box = detection[i];
            tmp_person.Box_scaled = rescaleROI(tmp_person.Box, 0.20);
            tmp_person.centre = (tmp_person.Box.tl() + tmp_person.Box.br()) / 2.0;
            tmp_person.story.push_back(tmp_person.centre);
            tmp_person.reid_descriptor = descriptors[i];
            tmp_person.hist_descriptor = evaluateHistogram(channels_HSV, tmp_person.Box_scaled);
            tmp_person.ID = -1;
            list_new.push_back(tmp_person);
        }
    }

    void PeopleList::generateCostMatrix()
    {
        std::cout << " cost matrix fattori" << std::endl;
        costMatrix.clear();
        for (int i = 0; i < list_new.size(); i++)
        {
            /* code */
            std::vector<double> vec_i;
            for (int j = 0; j < list_prev.size(); j++)
            {
                // similarity cost of reid net
                double cos_ij = (double)cosine_similarity(list_new[i].reid_descriptor, list_prev[j].reid_descriptor);
                if (list_prev[j].reid_descriptor_prev.size())
                {
                    double cos_ij_prev = (double)cosine_similarity(list_new[i].reid_descriptor, list_prev[j].reid_descriptor_prev);
                    cos_ij = std::max(cos_ij, cos_ij_prev);
                }

                // distance cost
                auto dd_dist = (list_new[i].Box.tl() + list_new[i].Box.br()) / 2.0 - (list_prev[j].Box.tl() + list_prev[j].Box.br()) / 2.0;
                double dist_ij = 6 * ((double)abs(dd_dist.x) / (double)img.cols + (double)abs(dd_dist.y) / (double)img.rows);
                // double dist_ij = sqrt(dd_dist.x * dd_dist.x + dd_dist.y * dd_dist.y) / (sqrt(img.cols * img.cols + img.rows * img.rows));

                // area cost
                double area_denom = std::max((double)list_prev[j].Box.area(), (double)list_new[i].Box.area());
                double c_area = abs((double)list_new[i].Box.area() - (double)list_prev[j].Box.area()) / area_denom;

                // histogram matching cost
                double hist_ij = acos((double)cosine_similarity(list_new[i].hist_descriptor[0], list_prev[j].hist_descriptor[0]));
                // hist_ij += acos((double)cosine_similarity(list_new[i].hist_descriptor[1], list_prev[j].hist_descriptor[1]));
                // hist_ij += acos((double)cosine_similarity(list_new[i].hist_descriptor[2], list_prev[j].hist_descriptor[2]));

                std::cout << "[ " << acos(cos_ij) / M_PI << ", " << hist_ij / M_PI << ", " << dist_ij << ", " << c_area << " ]  \t";

                // total cost
                double cost = ((acos(cos_ij) / M_PI + hist_ij / M_PI) + dist_ij + c_area) / 3;
                vec_i.push_back(cost);
            }
            std::cout << endl;
            costMatrix.push_back(vec_i);
        }
    }

    void PeopleList::doHungarian()
    {
        HungAssignment.clear();
        if (costMatrix.size() > 0)
        {
            // std::cout << "Solve Hungarian" << std::endl;
            double costHung = HungAlgo.Solve(costMatrix, HungAssignment);
        }
    }

    void PeopleList::sortNewDetection()
    {
        list = list_prev;
        std::vector<int> ind_new_used;
        // std::cout << " costMatrix " << std::endl;
        for (int i = 0; i < costMatrix.size(); i++)
        {
            /* code */
            for (int j = 0; j < costMatrix[0].size(); j++)
            {
                /* code */
                std::cout << costMatrix[i][j] << " ";
            }
            std::cout << endl;
        }

        for (int i = 0; i < HungAssignment.size(); i++)
        {
            /* code */
            std::cout << "assignment " << i << " --> " << HungAssignment[i];
            int ind_list = HungAssignment[i];
            if (ind_list != -1)
            {
                std::cout << "[ id " << list[ind_list].ID << " ]";
                double matchCost = costMatrix[i][ind_list];
                if (matchCost < 0.5)
                {
                    list[ind_list].Box = list_new[i].Box;
                    list[ind_list].counter_lost = 0;
                    list[ind_list].reid_descriptor_prev = list[i].reid_descriptor;
                    list[ind_list].reid_descriptor = list_new[i].reid_descriptor;
                    list[ind_list].story.push_back(list_new[i].centre);
                    ind_new_used.push_back(i);
                }
                else
                {
                    this->max_id++;
                    list_new[i].ID = max_id;
                    list.push_back(list_new[i]);
                }
            }
            else
            {
                this->max_id++;
                list_new[i].ID = max_id;
                list.push_back(list_new[i]);
            }
            std::cout << std::endl;
        }
    }

    void PeopleList::extractUser() {}

    void PeopleList::checkToKill()
    {
        // increase lost counter
        for (auto &p : list)
        {
            p.counter_lost++;
            if (p.ID == 1)
            {
                p.is_user = true;
            }
        }

        std::vector<Person> tmp_;
        for (int i = 0; i < list.size(); i++)
        {
            /* code */
            if (list[i].counter_lost < this->max_counter_lost | list[i].is_user)
                tmp_.push_back(list[i]);
        }
        list = tmp_;
    }

    void PeopleList::plotBoxes(cv::Mat &img)
    {

        for (int i = 0; i < detection.size(); i++)
        {
            /* code */
            cv::rectangle(img, detection[i], cv::Scalar(0, 158, 250), 4);
            cv::putText(img, std::to_string(i), detection[i].tl() + cv::Point2i(detection[i].width, 0), 1, 2, cv::Scalar(0, 158, 250)); //(img,std::to_string(p.ID),p.Box.tl());
        }
    }

    void PeopleList::plotTrack(cv::Mat &img)
    {
        for (auto &p : list)
        {
            /* code */
            auto colore = cv::Scalar(0, 0, 158);
            if (p.is_user)
                colore = cv::Scalar(0, 158, 0);
            cv::rectangle(img, p.Box, colore, 2);
            cv::putText(img, std::to_string(p.ID), p.Box.tl(), 1, 2, colore); //(img,std::to_string(p.ID),p.Box.tl());
        }
    }

    void PeopleList::plotStoryPosition(cv::Mat &img)
    {
        for (auto &p : list)
        {
            /* code */
            auto colore = cv::Scalar(0, 0, 158);
            if (p.is_user)
                colore = cv::Scalar(0, 158, 0);
            for (int i = 0; i < p.story.size() - 1; i++)
            {
                /* code */
                cv::line(img, p.story[i], p.story[i + 1], colore, 1);
            }
        }
    }

}