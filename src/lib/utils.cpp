#include "utils.h"

double cosine_similarity(std::vector<float> A, std::vector<float> B)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0u; i < A.size(); ++i)
    {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

std::vector<float> extract_face_feature(cv::dnn::Net net_reid, cv::Mat img)
{
    std::vector<float> feature;
    auto blob = cv::dnn::blobFromImage(img, 1.0f, cv::Size(128, 128), cv::Scalar(), true);
    net_reid.setInput(blob, "");
    auto outMat = net_reid.forward();
    float *data = (float *)outMat.data;
    for (size_t i = 0; i < outMat.total(); i++)
    {
        feature.push_back((float)(data[i]));
    }
    return feature;
}

std::vector<float> extract_body_feature(cv::dnn::Net *net_reid, cv::Mat img)
{
    // create blob & input
    cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 16.0f, cv::Size(256, 128), 0, false, false, CV_32F);

    net_reid->setInput(blob, "");

    // net execution
    cv::Mat outMat;
    net_reid->forward(outMat); // net_reid.forward();

    // extract data
    float *data = (float *)outMat.data;
    std::vector<float> tmp_feature;
    for (size_t i = 0; i < outMat.total(); i++)
    {
        tmp_feature.push_back((float)(data[i]));
    }
    return tmp_feature;
}

cv::Point getCentre_rect(cv::Rect ROI)
{
    cv::Point output;
    output.x = ROI.x + ROI.width / 2;
    output.y = ROI.y + ROI.height / 2;
    return output;
}

// resize rectangle that goes out of frame
void resizeRect(cv::Rect *oggetto, int cols, int rows)
{
    cv::Rect rectJ = (*oggetto);
    bool isChanged = false;

    int x = rectJ.x;
    int y = rectJ.y;
    int w = rectJ.width;
    int h = rectJ.height;

    // check sporgenza superiore
    if (x <= 0)
    {
        x = 1;
        isChanged = true;
    }
    // check sporgenza laterale
    if (y <= 0)
    {
        y = 1;
        isChanged = true;
    }
    if (x + w >= cols)
    {
        w = cols - x;
        isChanged = true;
    }
    if (y + h >= rows)
    {
        h = rows - y;
        isChanged = true;
    }

    if (isChanged == true)
        (*oggetto) = cv::Rect(x, y, w, h);
}

void resizeRects(std::vector<cv::Rect> *oggetti, int cols, int rows)
{
    for (int j = 0; j < oggetti->size(); j++)
    {
        resizeRect(&(*oggetti)[j], cols, rows);
    }
}

void rescaleRect(cv::Rect *rettangolo, float scala)
{
    cv::Rect tmp_Rect = (*rettangolo);
    cv::Point centro = getCentre_rect(tmp_Rect);
    int w = tmp_Rect.width * scala;
    int h = tmp_Rect.height * scala;
    int x = centro.x - w / 2;
    int y = centro.y - h / 2;

    (*rettangolo) = cv::Rect(x, y, w, h);
}

bool isOverlap(cv::Rect ROI_1, cv::Rect ROI_2, bool is_face)
{

    if (!is_face)
    {
        bool intersects = ((ROI_1 & ROI_2).area() > 0);
        bool centre1 = ROI_2.contains(getCentre_rect(ROI_1));
        bool centre2 = ROI_1.contains(getCentre_rect(ROI_2));
        return intersects & (centre1 | centre2);
    }
    else
    {
        /* code */
        bool TL = ROI_2.contains(ROI_1.tl());
        bool BR = ROI_2.contains(ROI_1.br());
        return TL & BR;
    }
}

float scoreOverlap(cv::Rect ROI_1, cv::Rect ROI_2)
{
    int area_1 = ROI_1.area();
    int area_2 = ROI_2.area();

    // std::cout << "areas a1: " << area_1 << "  a2: " << area_2 << std::endl;

    float area_max;
    float area_min;
    float score = 0.0;

    if (area_1 >= area_2)
    {
        area_max = (float)area_1;
        area_min = (float)area_2;
    }
    else
    {
        area_max = (float)area_2;
        area_min = (float)area_1;
    }
    float area_intersection = (ROI_1 & ROI_2).area();
    if (area_2 > 0)
    {
        /* code */
        score = (float)area_intersection / area_min;
    }

    return score;
}

bool findHisBody(std::vector<cv::Rect> bodies, cv::Rect face, cv::Rect *output)
{
    float out = false;
    for (int j = 0; j < bodies.size(); j++)
    {
        if (isOverlap(face, bodies[j], true))
        {
            // candidate_bodies.push_back(bodies[j]);
            (*output) = bodies[j];
            // personToTrack->gotBody = true;
            out = true;
            break;
        }
    }
    return out;
}

// Dection of the peoples on the frame, optional flag interessePosizione do "extractCentredRectangles"
std::vector<cv::Rect> DetectionPedestrian_on_frame(cv::Mat img, cv::dnn::Net *network)
{
    std::vector<cv::Rect> people_in_frame;
    float Confidence_Ped = 0.90;
    float blobscale_Pedestrian = 1.0;
    cv::Size sizeBlob_Pedestrian = cv::Size(672, 384);
    float scaleOut_Pedestrian = 1.0 / 255.0;
    cv::Scalar meanOut_Pedestrian = 0;
    float minimo_feature_match = 0.6;
    // create blob from image
    static cv::Mat blobFromImg_People;
    cv::dnn::blobFromImage(img, blobFromImg_People, blobscale_Pedestrian, sizeBlob_Pedestrian, false);

    //* Set blob as input of neural network.
    network->setInput(blobFromImg_People, "", scaleOut_Pedestrian, meanOut_Pedestrian);

    // compure network
    auto outMat_People = network->forward();
    // extract object from result of network
    auto data = (float *)outMat_People.data;

    // extract data
    for (size_t i = 0; i < outMat_People.total(); i += 7)
    {
        float confidence = data[i + 2];
        if (confidence > Confidence_Ped)
        {
            int left = (int)(data[i + 3] * img.cols);
            int top = (int)(data[i + 4] * img.rows);
            int right = (int)(data[i + 5] * img.cols);
            int bottom = (int)(data[i + 6] * img.rows);
            int width = right - left + 1;
            int height = bottom - top + 1;

            people_in_frame.push_back(cv::Rect(left, top, width, height));
        }
    }
    return people_in_frame;
}

// Detection of faces on the frame
std::vector<cv::Rect> DetectionFaces_on_frame(cv::Mat img, cv::dnn::Net *network)
{
    std::vector<cv::Rect> Faces_on_frame;
    float Confidence_face = 0.90;
    float blobscale_Face = 1.0;
    cv::Size sizeBlob_Face = cv::Size(300, 300);
    float scaleOut_Face = 1.0 / 255.0;
    cv::Scalar meanOut_Face = 0;

    // create blob from image
    static cv::Mat blobFromImg;
    cv::dnn::blobFromImage(img, blobFromImg, blobscale_Face, sizeBlob_Face, false);

    //* Set blob as input of neural network.
    network->setInput(blobFromImg, "", scaleOut_Face, meanOut_Face);

    // compure network
    auto outMat = network->forward();
    // extract object from result of network
    auto data = (float *)outMat.data;

    // extract data
    for (size_t i = 0; i < outMat.total(); i += 7)
    {
        float confidence = data[i + 2];
        if (confidence > Confidence_face)
        {
            int left = (int)(data[i + 3] * img.cols);
            int top = (int)(data[i + 4] * img.rows);
            int right = (int)(data[i + 5] * img.cols);
            int bottom = (int)(data[i + 6] * img.rows);
            int width = right - left + 1;
            int height = bottom - top + 1;

            Faces_on_frame.push_back(cv::Rect(left, top, width, height));
        }
    }
    return Faces_on_frame;
}

cv::Rect VecToRect(const std::vector<float> &vec)
{
    return cv::Rect(cv::Point(vec[0], vec[1]), cv::Point(vec[2], vec[3]));
}

void DrawRectangles(cv::Mat &img,
                    const std::vector<std::vector<float>> &vecVecFloat)
{
    for (const auto &vec : vecVecFloat)
        cv::rectangle(img, VecToRect(vec), WHITE_COLOR);
}

void DrawRectangles(cv::Mat &img,
                    const std::vector<cv::Rect> &vecRect)
{
    for (const auto &rect : vecRect)
        cv::rectangle(img, rect, WHITE_COLOR);
}
