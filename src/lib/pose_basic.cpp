#include "./pose_basic.h"
#include "./pose_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs.hpp>
// #include <tbb/parallel_for.h>

//   __                  _    _
//  / _| _  _  _ _   __ | |_ (_) ___  _ _   ___
// |  _|| || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

void extract_all_keypoints(cv::Mat result, cv::Size targetSize,
                           vector<vector<KeyPoint_pose>> *all_keyPoints,
                           std::vector<KeyPoint_pose> *keyPointsList, float thresh)
{

    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;
    // float thresh = 0.10;
    int keyPointId = 0;
    // cv::Size blur_size = cv::Size(3, 3);
    // vector<vector<KeyPoint_pose>> all_keyPoints[18];
    // find the position of the body parts
    for (int n = 0; n < nparts; n++)
    {
        vector<KeyPoint_pose> keyPoints;
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        cv::UMat UheatMap;
        heatMap.copyTo(UheatMap);
        // cv::UMat UresizedHeatMap;
        // cv::resize(UheatMap, UresizedHeatMap, targetSize);
        cv::resize(UheatMap, UheatMap, targetSize);

        // cv::UMat UsmoothProbMap;
        // cv::GaussianBlur(UresizedHeatMap, UsmoothProbMap, blur_size, 0, 0);
        // cv::GaussianBlur(UheatMap, UheatMap, blur_size, 0, 0);
        UheatMap.copyTo(heatMap);

        // cv::Mat smoothProbMap;
        // UsmoothProbMap.copyTo(smoothProbMap);

        cv::UMat UmaskedProbMap;
        // cv::threshold(UsmoothProbMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        cv::threshold(UheatMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        // cv::Mat maskedProbMap;
        // UmaskedProbMap.copyTo(maskedProbMap);

        // maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);
        UmaskedProbMap.convertTo(UmaskedProbMap, CV_8U, 1);

        std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::findContours(UmaskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); ++i)
        {
            // cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());
            cv::UMat blobMask = cv::UMat::zeros(UheatMap.rows, UheatMap.cols, UheatMap.type());

            cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

            double maxVal;
            cv::Point maxLoc;

            // cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);
            cv::minMaxLoc(UheatMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

            keyPoints.push_back(KeyPoint_pose(maxLoc, heatMap.at<float>(maxLoc.y, maxLoc.x)));
        }

        // give a unic id number to all points
        for (int i = 0; i < keyPoints.size(); i++, keyPointId++)
        {
            keyPoints[i].id = keyPointId;
        }

        all_keyPoints->push_back(keyPoints);
        keyPointsList->insert(keyPointsList->end(), keyPoints.begin(), keyPoints.end());
    };
};

KeyPoint_pose keypoint_from_contour(cv::Mat heatMap, std::vector<cv::Point> contour, cv::Mat *img)
{
    cv::Mat covMat;
    KeyPoint_pose keypoint = KeyPoint_pose(cv::Point(-1, -1), 0);
    cv::RotatedRect ellipse = generate_cov_ellipse(heatMap, contour, &covMat);
    // cout <<"cov_mat point extraction : " << covMat <<endl;
    // cout << ellipse.size << endl;
    int x_mean = ellipse.center.x;
    int y_mean = ellipse.center.y;

    if (x_mean > 0 & x_mean<img->size().width & y_mean> 0 & y_mean < img->size().height)
    {
        // punti.push_back(cv::Point((int)x_mean, (int)y_mean));
        KeyPoint_pose tmp_point = KeyPoint_pose(cv::Point((int)x_mean, (int)y_mean), heatMap.at<double>((int)x_mean, (int)y_mean));
        tmp_point.ellipse = ellipse;
        tmp_point.covMatrix = covMat;
        keypoint = tmp_point;
    }
    // else
    // {
    //     keypoint = KeyPoint_pose(cv::Point(-1, -1), 0);
    //     // cout << "excluded point" << excluded_points << "  " << cv::Point((int)x_mean, (int)y_mean) << "\n";
    // }
    return keypoint;
}

void extract_all_keypoints_cov(cv::Mat result, cv::Size targetSize,
                               vector<vector<KeyPoint_pose>> *all_keyPoints,
                               std::vector<KeyPoint_pose> *keyPointsList, cv::Mat *img, bool plot_ellipse, bool plot_circle, float thresh)
{
    cv::Rect img_rect = cv::Rect(0, 0, targetSize.width, targetSize.height);
    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;
    // float thresh = 0.10;
    int keyPointId = 0;
    int excluded_points = 0;
    // cv::Size blur_size = cv::Size(3, 3);
    // cv::Size blur_size = cv::Size(5, 5);
    // vector<vector<KeyPoint_pose>> all_keyPoints[18];
    // find the position of the body parts
    std::vector<RotatedRect> ellissi;
    std::vector<Point> punti;
    for (int n = 0; n < nparts; n++)
    {
        vector<KeyPoint_pose> keyPoints;
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        cv::UMat UheatMap;
        heatMap.copyTo(UheatMap);
        // cv::UMat UresizedHeatMap;
        cv::resize(UheatMap, UheatMap, targetSize, 0, 0, cv::INTER_LINEAR);

        // cv::UMat UsmoothProbMap;
        // cv::GaussianBlur(UresizedHeatMap, UsmoothProbMap, blur_size, 0, 0);
        cv::Mat smoothProbMap;
        UheatMap.copyTo(heatMap);

        cv::UMat UmaskedProbMap;
        cv::threshold(UheatMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        // cv::Mat maskedProbMap;
        // UmaskedProbMap.copyTo(maskedProbMap);

        UmaskedProbMap.convertTo(UmaskedProbMap, CV_8U, 1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(UmaskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // for (int i = 0; i < contours.size(); i++)
        // {
        //     cv::Mat covMat;
        //     cv::RotatedRect ellipse = generate_cov_ellipse(heatMap, contours[i], &covMat);
        //     // cout <<"cov_mat point extraction : " << covMat <<endl;
        //     // cout << ellipse.size << endl;
        //     int x_mean = ellipse.center.x;
        //     int y_mean = ellipse.center.y;

        //     if (x_mean > 0 & x_mean<img->size().width & y_mean> 0 & y_mean < img->size().height)
        //     {
        //         punti.push_back(cv::Point((int)x_mean, (int)y_mean));
        //         KeyPoint_pose tmp_point = KeyPoint_pose(cv::Point((int)x_mean, (int)y_mean), heatMap.at<double>((int)x_mean, (int)y_mean));
        //         tmp_point.ellipse = ellipse;
        //         tmp_point.covMatrix = covMat;
        //         keyPoints.push_back(tmp_point);
        //     }
        //     else
        //     {
        //         excluded_points++;
        //         cout << "excluded point" << excluded_points << "  " << cv::Point((int)x_mean, (int)y_mean) << "\n";
        //     }
        // }
        auto all_keypoint_tmp = std::vector<KeyPoint_pose>(contours.size(),KeyPoint_pose(cv::Point(-1,-1),0));
        for (int i = 0; i < contours.size(); i++)
        {
            all_keypoint_tmp[i] = keypoint_from_contour(heatMap, contours[i], img);
        }

        // parallel for loop
        // tbb::parallel_for(tbb::blocked_range<int>(0, all_keypoint_tmp.size()),
        //                   [&](tbb::blocked_range<int> r) {
        //                       for (int i = r.begin(); i < r.end(); ++i)
        //                       {
        //                         all_keypoint_tmp[i] = keypoint_from_contour(heatMap, contours[i], img);
        //                       }
        //                   });

        // TODO: merge points inside the same ellipse ? keeping the high probable ?
        keyPoints = all_keypoint_tmp;

        // give a unic id number to all points
        for (int i = 0; i < keyPoints.size(); i++, keyPointId++)
        {
            keyPoints[i].id = keyPointId;
        }

        all_keyPoints->push_back(keyPoints);
        keyPointsList->insert(keyPointsList->end(), keyPoints.begin(), keyPoints.end());
    };

    //* plot for testing
    if (plot_ellipse)
    {
        for (int i = 0; i < ellissi.size(); i++)
        {
            try
            {
                cv::ellipse((*img), ellissi[i], cv::Scalar(255, 255, 0), 1, 8);
            }
            catch (cv::Exception &e)
            {
            }
        }
    }
    if (plot_circle)
    {
        for (int i = 0; i < punti.size(); i++)
        {
            cv::circle((*img), punti[i], 3, cv::Scalar(255, 255, 0), 1, 8);
        }
    }
    // cv::imshow("test ellissi", img);
    // int k = waitKey(10);
    // if (k == 32) // space = pause
    // {
    //     waitKey(0);
    // }
};

//_______________________________________________________________________________________
//____________ pairwise functions _______________________________________________________
void populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints, std::vector<cv::Point> &interpCoords)
{
    float xStep = ((float)(b.x - a.x)) / (float)(numPoints - 1);
    float yStep = ((float)(b.y - a.y)) / (float)(numPoints - 1);

    interpCoords.push_back(a);

    for (int i = 1; i < numPoints - 1; ++i)
    {
        interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
    }

    interpCoords.push_back(b);
}

vector<cv::Mat> getValidPairs(cv::Mat paf_blob, cv::Size img_size,
                              const std::vector<std::vector<KeyPoint_pose>> detectedKeypoints,
                              std::vector<std::vector<ValidPair>> *validPairs,
                              std::set<int> *invalidPairs)
{
    cv::Rect img_rect = cv::Rect(cv::Point(), img_size);

    // from learn opencv:
    int nInterpSamples = 10;
    float pafScoreTh = 0.1;
    float confTh = 0.70;

    // my tests
    // int nInterpSamples = 10;
    // float pafScoreTh = 0.10;
    // float confTh = 0.7;

    // int nInterpSamples = 10;
    // float pafScoreTh = 0.1;
    // float confTh = 0.45;

    // int nInterpSamples = 20;
    // float pafScoreTh = 0.1;
    // float confTh = 0.70;
    // int nInterpSamples = 100;
    // float pafScoreTh = 0.2;
    // float confTh = 0.5;

    int H = paf_blob.size[2];
    int W = paf_blob.size[3];

    // cv::Mat PafA_total = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);
    // cv::Mat PafB_total = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);
    cv::Mat zeros = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);

    for (int k = 0; k < mapIdx.size(); ++k)
    {
        // cout << "k  " << k << endl;
        //A->B constitute a limb
        Mat pafA(H, W, CV_32F, paf_blob.ptr(0, mapIdx[k].first - 19));
        cv::resize(pafA, pafA, img_size, 0, 0, cv::INTER_CUBIC);

        Mat pafB(H, W, CV_32F, paf_blob.ptr(0, mapIdx[k].second - 19));
        cv::resize(pafB, pafB, img_size, 0, 0, cv::INTER_CUBIC);

        //Find the keypoints for the first and second limb
        const std::vector<KeyPoint_pose> &candA = detectedKeypoints[posePairs[k].first];
        const std::vector<KeyPoint_pose> &candB = detectedKeypoints[posePairs[k].second];

        int nA = candA.size();
        int nB = candB.size();

        // add(pafA * 200, zeros, PafA_total);
        // add(pafB * 200, zeros, PafB_total);

        /*
		  # If keypoints for the joint-pair is detected
		  # check every joint in candA with every joint in candB
		  # Calculate the distance vector between the two joints
		  # Find the PAF values at a set of interpolated points between the joints
		  # Use the above formula to compute a score to mark the connection valid
		*/

        if (nA != 0 && nB != 0)
        {
            std::vector<ValidPair> localValidPairs;

            for (int i = 0; i < nA; ++i)
            {
                if (!img_rect.contains(candA[i].point))
                {
                    continue;
                }

                int maxJ = -1;
                float maxScore = -1;
                bool found = false;

                for (int j = 0; j < nB; ++j)
                {
                    if (!img_rect.contains(candB[j].point))
                    {
                        continue;
                    }
                    // calculate distance between cand A & cand B
                    std::pair<float, float> distance(candB[j].point.x - candA[i].point.x, candB[j].point.y - candA[i].point.y);
                    float norm = std::sqrt(distance.first * distance.first + distance.second * distance.second);

                    if (!norm)
                    {
                        continue;
                    }

                    // get unitary versor
                    distance.first /= norm;
                    distance.second /= norm;

                    //Find p(u)
                    std::vector<cv::Point> interpCoords;
                    populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);
                    //Find L(p(u))
                    std::vector<std::pair<float, float>> pafInterp;
                    for (int l = 0; l < interpCoords.size(); ++l)
                    {
                        pafInterp.push_back(
                            std::pair<float, float>(
                                pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
                                pafB.at<float>(interpCoords[l].y, interpCoords[l].x)));
                    }

                    std::vector<float> pafScores;
                    float sumOfPafScores = 0;
                    int numOverTh = 0;
                    for (int l = 0; l < pafInterp.size(); ++l)
                    {
                        float score = abs(pafInterp[l].first * distance.first + pafInterp[l].second * distance.second);
                        sumOfPafScores += score;
                        if (score > pafScoreTh)
                        {
                            ++numOverTh;
                        }

                        pafScores.push_back(score);
                    }

                    float avgPafScore = sumOfPafScores / ((float)pafInterp.size());

                    if (((float)numOverTh) / ((float)nInterpSamples) > confTh)
                    {
                        if (avgPafScore > maxScore)
                        {
                            maxJ = j;
                            maxScore = avgPafScore;
                            found = true;
                        }
                    }

                } /* j */

                if (found)
                {
                    localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
                }

            } /* i */

            validPairs->push_back(localValidPairs);
        }
        else
        {
            invalidPairs->insert(k);
            validPairs->push_back(std::vector<ValidPair>());
        }
    } /* k */
    vector<Mat> output;
    // output.push_back(PafA_total);
    // output.push_back(PafB_total);
    return output;
}

void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                            const std::set<int> &invalidPairs,
                            std::vector<std::vector<int>> &personwiseKeypoints)
{
    for (int k = 0; k < mapIdx.size(); ++k)
    {
        if (invalidPairs.find(k) != invalidPairs.end())
        {
            continue;
        }

        const std::vector<ValidPair> &localValidPairs(validPairs[k]);

        int indexA(posePairs[k].first);
        int indexB(posePairs[k].second);

        for (int i = 0; i < localValidPairs.size(); ++i)
        {
            bool found = false;
            int personIdx = -1;

            for (int j = 0; !found && j < personwiseKeypoints.size(); ++j)
            {
                if (indexA < personwiseKeypoints[j].size() &&
                    personwiseKeypoints[j][indexA] == localValidPairs[i].aId)
                {
                    personIdx = j;
                    found = true;
                }
            } /* j */

            if (found)
            {
                personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
            }
            else if (k < 17)
            {
                std::vector<int> lpkp(std::vector<int>(18, -1));

                lpkp.at(indexA) = localValidPairs[i].aId;
                lpkp.at(indexB) = localValidPairs[i].bId;

                personwiseKeypoints.push_back(lpkp);
            }

        } /* i */
    }     /* k */
}

void plot_all_skeleton(cv::Mat *img, std::vector<std::vector<int>> personwiseKeypoints,
                       std::vector<KeyPoint_pose> keyPointsList, bool white, cv::Scalar color)
{

    for (int i = 0; i < 17; i++)
    {
        for (int n = 0; n < personwiseKeypoints.size(); ++n)
        {
            auto pt_cl = (float)(n + 2) / (float)(personwiseKeypoints.size() + 2);
            const std::pair<int, int> &posePair = posePairs[i];
            int indexA = personwiseKeypoints[n][posePair.first];
            int indexB = personwiseKeypoints[n][posePair.second];

            if (indexA == -1 || indexB == -1)
            {
                continue;
            }

            KeyPoint_pose &kpA = keyPointsList[indexA];
            KeyPoint_pose &kpB = keyPointsList[indexB];

            if (white)
            {
                if (color != cv::Scalar(0, 0, 0))
                {
                    cv::line((*img), kpA.point, kpB.point, cv::Scalar(color[0] * pt_cl, color[1] * pt_cl, color[2] * pt_cl), 1, cv::LINE_AA);
                }
                else
                    cv::line((*img), kpA.point, kpB.point, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
            else
            {
                cv::line((*img), kpA.point, kpB.point, colors_left_right[i], 1, cv::LINE_AA);
            }
        }
    }
}

void plot_skeleton(cv::Mat *img, std::vector<int> personwiseKeypoints,
                   std::vector<KeyPoint_pose> keyPointsList, bool white)
{

    for (int i = 0; i < 17; i++)
    {

        const std::pair<int, int> &posePair = posePairs[i];
        int indexA = personwiseKeypoints[posePair.first];
        int indexB = personwiseKeypoints[posePair.second];

        if (indexA == -1 || indexB == -1)
        {
            continue;
        }

        KeyPoint_pose &kpA = keyPointsList[indexA];
        KeyPoint_pose &kpB = keyPointsList[indexB];

        if (white)
        {
            cv::line((*img), kpA.point, kpB.point, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
        }
        else
        {
            cv::line((*img), kpA.point, kpB.point, colors_left_right[i], 3, cv::LINE_AA);
        }
    }
}

void plot_skeleton_listpoint(cv::Mat *img, std::vector<Point> list_point, Scalar color)
{

    for (int i = 0; i < 17; i++)
    {

        const std::pair<int, int> &posePair = posePairs[i];
        Point kpA = list_point[posePair.first];
        Point kpB = list_point[posePair.second];

        if (kpA.x == -1 || kpB.x == -1)
        {
            continue;
        }

        cv::line((*img), kpA, kpB, color, 1, cv::LINE_AA);
    }
}

std::vector<std::vector<int>> good_skeleton_extraction(std::vector<std::vector<int>> personwiseKeypoints, int min_pairs_th)
{
    std::vector<std::vector<int>> good_skeleton;
    int counter_pairs = 0;
    for (int n = 0; n < personwiseKeypoints.size(); ++n)
    {
        for (int i = 0; i < 17; i++)
        {
            const std::pair<int, int> &posePair = posePairs[i];
            int indexA = personwiseKeypoints[n][posePair.first];
            int indexB = personwiseKeypoints[n][posePair.second];

            if (indexA == -1 || indexB == -1)
            {
                continue;
            }
            else
            {
                counter_pairs++;
            }
        }
        if (counter_pairs > min_pairs_th)
        {
            good_skeleton.push_back(personwiseKeypoints[n]);
            counter_pairs = 0;
        }
        else
        {
            counter_pairs = 0;
        }
    }
    return good_skeleton;
}

// for analysis
void save_heatmap(cv::Mat result, cv::Size targetSize, string path_to_save)
{

    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;

    for (int n = 0; n < nparts; n++)
    {
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        cv::resize(heatMap, heatMap, targetSize);
        heatMap.convertTo(heatMap, CV_8U, 255.0);

        string path = path_to_save;
        string fullpath = path + to_string(n) + ".png";
        std::cout << fullpath << endl;

        cv::imwrite(fullpath, heatMap);

        // cv::imwrite(fullpath, heatMap);
    }
}