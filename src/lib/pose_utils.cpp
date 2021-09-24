#include "pose_utils.h"
using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

#define PI 3.14159265

//  _                          _        _                _    _  _
// | |__ ___  _  _  _ __  ___ (_) _ _  | |_  ___   _  _ | |_ (_)| | ___
// | / // -_)| || || '_ \/ _ \| || ' \ |  _|(_-<  | || ||  _|| || |(_-<
// |_\_\\___| \_, || .__/\___/|_||_||_| \__|/__/   \_,_| \__||_||_|/__/
//            |__/ |_|
//
bool ccw(cv::Point A, cv::Point B, cv::Point C)
{
    /* Returns True if the 3 points A,
        B and C are listed in a counterclockwise order
        ie if the slope of the line AB is less than the slope of AC
        https : //bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    */
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

bool intersect(cv::Point A, cv::Point B, cv::Point C, cv::Point D)
{
    /*
        Return true if line segments AB and CD intersect
        https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    */
    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0 or C.x < 0 or C.y < 0 or D.x < 0 or D.y < 0)
    {
        return false;
    }
    else
    {
        return ccw(A, C, D) != ccw(B, C, D) & ccw(A, B, C) != ccw(A, B, D);
    }
}

float angle(cv::Point A, cv::Point B, cv::Point C)
{
    //* Calculate the angle between segment(A,B) and segment (B,C)

    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0 or C.x < 0 or C.y < 0)
    {
        std::cout << "error input" << std::endl;
        return 0;
    }
    return (atan2(C.y - B.y, C.x - B.x) - atan2(A.y - B.y, A.x - B.x)) * 180 / PI;
}

float vertical_angle(cv::Point A, cv::Point B)
{
    //* Calculate the angle between segment(A,B) and vertical axe
    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0)
    {
        std::cout << "error input" << std::endl;
        return 0;
    }
    else
    {
        return (atan2(B.y - A.y, B.x - A.x) - PI / 2.0) * 180 / PI;
    }
}

float sq_distance(cv::Point A, cv::Point B)
{

    // Calculate the square of the distance between points A and B
    return (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
}

float distance(cv::Point A, cv::Point B)
{

    // Calculate the distance between points A and B
    return sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
}

long double distanceMahalanobis(cv::Point A, cv::Point B_cov, cv::Mat covMat)
{
    // cout << "cov inverse " << covMat.inv() << endl;
    Mat out = (Mat((Point2f)(A - B_cov))).t() * (covMat.inv()) * Mat((Point2f)(A - B_cov));
    // cout << "Matrix out" <<  out << endl;
    long double square_val = out.at<double>(0, 0);
    // cout << sqrt(square_val) << endl;
    return (long double)sqrt(square_val);
}

//  ___                  _    _
// | __| _  _  _ _   __ | |_ (_) ___  _ _   ___
// | _| | || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

cv::RotatedRect getErrorEllipse(float chisquare_val, cv::Point2f mean, cv::Mat covmat)
{

    //Get the eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covmat, eigenvalues, eigenvectors);
    // cout << "eigen values : \n" << eigenvalues.at<float>(0) << "\n" << eigenvalues.at<float>(1) << endl;

    //Calculate the angle between the largest eigenvector and the x-axis
    float angle = atan2(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(0, 0));

    //Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if (angle < 0)
        angle += 6.28318530718;

    //Conver to degrees instead of radians
    angle = 180 * angle / 3.14159265359;

    //Calculate the size of the minor and major axes
    float halfmajoraxissize = chisquare_val * sqrt(eigenvalues.at<float>(0));
    float halfminoraxissize = chisquare_val * sqrt(eigenvalues.at<float>(1));

    //Return the oriented ellipse
    //The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
    return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);
}

cv::RotatedRect generate_cov_ellipse(cv::Mat heatMap, vector<Point> contour, cv::Mat *covMat)
{
    cv::RotatedRect ellipse;

    cv::Rect contour_rect = cv::boundingRect(contour);

    if (contour_rect.empty())
    {
        return ellipse;
    }

    // evalaute the means value for each row and col
    std::vector<int> vector_x;
    std::vector<int> vector_y;
    std::vector<float> vector_p;
    for (int m = contour_rect.tl().x; m < contour_rect.tl().x + contour_rect.br().x; m++)
    {
        for (int n = contour_rect.tl().y; n < contour_rect.tl().y + contour_rect.br().y; n++)
        {
            if (cv::pointPolygonTest(contour, cv::Point(m, n), false) == 0)
            {
                vector_x.push_back(m);
                vector_y.push_back(n);
                vector_p.push_back(heatMap.at<float>(n,m));
            }
        }
    }

    // expected value
    float x_mean = 0.0;
    float y_mean = 0.0;
    float p_tot = accumulate(vector_p.begin(), vector_p.end(), 0.0);
    for (int k = 0; k < vector_x.size(); k++)
    {
        x_mean += ((vector_x[k] * vector_p[k]));
        y_mean += ((vector_y[k] * vector_p[k]));
        // p_tot += vector_p[k];
    }
    x_mean /= p_tot;
    y_mean /= p_tot;

    // evaluate covariance
    float cov_xx = 0;
    float cov_yy = 0;
    float cov_xy = 0;
    for (int k = 0; k < vector_x.size(); k++)
    {
        cov_xx += (vector_x[k] - x_mean) * (vector_x[k] - x_mean) * vector_p[k];
        cov_yy += (vector_y[k] - y_mean) * (vector_y[k] - y_mean) * vector_p[k];
        cov_xy += (vector_x[k] - x_mean) * (vector_y[k] - y_mean) * vector_p[k];
    }
    cov_xx /= p_tot;
    cov_yy /= p_tot;
    cov_xy /= p_tot;
    // covariance matrix
    float data[4] = {cov_xx, cov_xy, cov_xy, cov_yy};
    cv::Mat cov_matrix = cv::Mat(2, 2, CV_32F, data);
    // std::cout << " - cov Matrix : " << cov_matrix << std::endl;
    // std::cout << "Point mean " << cv::Point(x_mean, y_mean) << std::endl;

    //Calculate the error ellipse for a 95% confidence interval ( 2.4477 )
    cov_matrix.copyTo((*covMat));
    ellipse = getErrorEllipse(2.4477, cv::Point(x_mean, y_mean), cov_matrix);
    return ellipse;
}

void text_select_key(int number)
{
    if (number == 0)
        cout << "NOSE";
    else if (number == 1)
        cout << "NECK";
    else if (number == 2)
        cout << "RIGHT_SHOULDER";
    else if (number == 3)
        cout << "RIGHT_ELBOW";
    else if (number == 4)
        cout << "RIGHT_WRIST";
    else if (number == 5)
        cout << "LEFT_SHOULDER";
    else if (number == 6)
        cout << "LEFT_ELBOW";
    else if (number == 7)
        cout << "LEFT_WRIST";
    else if (number == 8)
        cout << "RIGHT_HIP";
    else if (number == 9)
        cout << "RIGHT_KNEE";
    else if (number == 10)
        cout << "RIGHT_ANKLE";
    else if (number == 11)
        cout << "LEFT_HIP";
    else if (number == 12)
        cout << "LEFT_KNEE";
    else if (number == 13)
        cout << "LEFT_ANKLE";
    else if (number == 14)
        cout << "RIGHT_EYE";
    else if (number == 15)
        cout << "RIGHT_EAR";
    else if (number == 16)
        cout << "LEFT_EYE";
    else if (number == 17)
        cout << "LEFT_EAR";
}

Point mouse_pt = Point(-1, -1);
bool newCoords = false;

void mouse_callback(int event, int x, int y, int flag, void *param)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        // Store point coordinates
        mouse_pt.x = x;
        mouse_pt.y = y;
        newCoords = true;
    }
}

void get_point_user(Mat *img_get_point, vector<Point> *clicked_point)
{
    namedWindow("get_points", 1);
    Mat tmp_img;
    img_get_point->copyTo(tmp_img);
    // imshow("get_points", tmp_img);

    // Point mouse_pt(-1, -1);
    for (int count_point = 0; count_point < 18; count_point++)
    {
        /* code */
        bool next_point = false;
        cout << "select the point ";
        text_select_key(count_point);
        cout << endl;

        while (next_point == false)
        {

            img_get_point->copyTo(tmp_img);
            while (mouse_pt.x == -1 & mouse_pt.y == -1)
            {
                imshow("get_points", tmp_img);
                setMouseCallback("get_points", mouse_callback);
                waitKey(100);
            }
            destroyAllWindows();
            // cout << "seleted point " << mouse_pt << endl;
            circle(tmp_img, mouse_pt, 2, Scalar(255, 255, 0), 2, 8, 0);
            imshow("get_points", tmp_img);
            cout << "The selected point is ok [o] / redo [r] / notvisible [n]" << endl;
            int key = waitKey(0);
            if (key == 111) // O -> Ok, next point
            {
                clicked_point->push_back(mouse_pt);
                cout << "saved point " << count_point << " >> " << mouse_pt << endl;
                next_point = true;
                mouse_pt.x = -1;
                mouse_pt.y = -1;
            }
            else if (key == 114) // R -> reselect point
            {
                cout << "select new point " << count_point << endl;
                mouse_pt.x = -1;
                mouse_pt.y = -1;
                next_point = false;
            }
            else if (key == 110) // n -> point not visible
            {
                mouse_pt.x = -1;
                mouse_pt.y = -1;
                clicked_point->push_back(mouse_pt);
                cout << "saved point " << count_point << " >> " << mouse_pt << endl;
                next_point = true;
            }
            else if (key == 27) // Esc -> annulla selezione
            {
                count_point = 20;
                for (int l = clicked_point->size(); l < clicked_point->size() - 18; l++)
                {
                    clicked_point->push_back(Point(-1, -1));
                }
                next_point = true;
            }
        }
    }
    destroyWindow("get_points");
}

cv::Rect get_bounding_rect_for_pose(std::vector<int> pose, std::vector<KeyPoint_pose> keyPointsList)
{
    cv::Rect boundingRect;
    std::vector<int> list_x;
    std::vector<int> list_y;
    for (int i = 0; i < pose.size(); i++)
    {
        int indx = pose[i];
        if (indx == -1)
            continue;

        KeyPoint_pose &kp = keyPointsList[indx];
        list_x.push_back(kp.point.x);
        list_y.push_back(kp.point.y);
    }
    int x_min = *min_element(list_x.begin(), list_x.end());
    int y_min = *min_element(list_y.begin(), list_y.end());
    int x_max = *max_element(list_x.begin(), list_x.end());
    int y_max = *max_element(list_y.begin(), list_y.end());

    boundingRect = cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max));

    return boundingRect;
}

cv::Rect get_bounding_rect_for_listpose(std::vector<cv::Point> pose)
{
    cv::Rect boundingRect;
    std::vector<int> list_x;
    std::vector<int> list_y;
    for (int i = 0; i < pose.size(); i++)
    {
        Point kp = pose[i];
        if (kp.x == -1 | kp.y == -1)
            continue;

        list_x.push_back(kp.x);
        list_y.push_back(kp.y);
    }
    int x_min = *min_element(list_x.begin(), list_x.end());
    int y_min = *min_element(list_y.begin(), list_y.end());
    int x_max = *max_element(list_x.begin(), list_x.end());
    int y_max = *max_element(list_y.begin(), list_y.end());

    boundingRect = cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max));

    return boundingRect;
};

vector<int> generate_match_poses(std::vector<std::vector<int>> pose_1, std::vector<KeyPoint_pose> keyPointsList_1,
                                 std::vector<std::vector<int>> pose_2, std::vector<KeyPoint_pose> keyPointsList_2,
                                 Mat *display_img, bool print_box, bool print_assignment)
{

    vector<vector<double>> costMatrix;
    if (pose_1.size() > 0)
    {
        for (int j = 0; j < pose_2.size(); j++)
        {
            vector<double> tmp_row;
            Rect pose_1_j = get_bounding_rect_for_pose(pose_1[j], keyPointsList_1);
            if (print_box)
            {
                cv::rectangle((*display_img), pose_1_j, Scalar(0, 200, 0));
                cv::putText((*display_img), to_string(j), pose_1_j.tl(), 0.5, 0.5, Scalar(0, 200, 0));
            }
            for (int k = 0; k < pose_2.size(); k++)
            {
                Rect pose_2_k = get_bounding_rect_for_pose(pose_2[k], keyPointsList_2);
                if (print_box & j == 0)
                {
                    cv::rectangle((*display_img), pose_2_k, Scalar(200, 0, 0));
                    cv::putText((*display_img), to_string(k), pose_2_k.tl(), 0.5, 0.5, Scalar(200, 0, 0));
                }
                double tmp_overlap = 1.0 / scoreOverlap(pose_1_j, pose_2_k);
                tmp_row.push_back(tmp_overlap);
            }
            costMatrix.push_back(tmp_row);
        }
    }

    HungarianAlgorithm HA;
    vector<int> assignment;
    double cost = HA.Solve(costMatrix, assignment);
    if (print_assignment)
    {
        std::cout << "Pose assignment ";
        for (int i = 0; i < assignment.size(); i++)
        {
            if (print_assignment)
                if (assignment[i] >= 0)
                    std::cout << i << "-" << assignment[i] << " ; ";
                else
                    std::cout << "n ; ";
        }
        std::cout << endl;
    }
    return assignment;
}

void save_point_and_distances(vector<int> assignment, vector<int> *euclidean_distances, vector<int> *mahalanobis_distances,
                              vector<vector<int>> poses_peak, vector<KeyPoint_pose> keyPointsList_peak,
                              vector<vector<int>> poses_cov, vector<KeyPoint_pose> keyPointsList_cov,
                              string folder, bool save_to_file)
{

    // Export points and distance to file
    mahalanobis_distances->clear();
    euclidean_distances->clear();

    ostringstream PP_string;
    ostringstream PM_string;
    ostringstream Distance;

    // Write to the file
    for (int i = 0; i < assignment.size(); i++)
    {
        if (assignment[i] >= 0)
        {
            for (int k = 0; k < poses_peak[i].size(); k++)
            {
                Point kcc, kpp;
                Mat cov_mat;
                int index = poses_peak[i][k];
                if (index == -1)
                {
                    /* code */
                    if (save_to_file)
                        PP_string <<"-1,-1\n";
                }
                else
                {
                    KeyPoint_pose kp = keyPointsList_peak[index];
                    kpp = kp.point;
                    PP_string << kp.point.x << "," << kp.point.y << "\n";
                }

                int index_cov = poses_cov[assignment[i]][k];
                if (index_cov == -1)
                {
                    /* code */
                    PM_string << "-1,-1\n";
                }
                else
                {
                    KeyPoint_pose kc = keyPointsList_cov[index_cov];
                    kcc = kc.point;
                    cov_mat = kc.covMatrix;
                    PM_string << kc.point.x << "," << kc.point.y << "\n";
                }

                if (index == -1 | index_cov == -1)
                {
                    Distance << "inf,inf\n";
                }
                else
                {
                    int euclidean_distance = distance(kcc, kpp);
                    euclidean_distances->push_back(euclidean_distance);
                    Distance << euclidean_distance << ",";
                    try
                    {
                        /* code */
                        float maha_dist = distanceMahalanobis(kpp, kcc, cov_mat);
                        mahalanobis_distances->push_back(maha_dist);
                        Distance << maha_dist;
                        Distance << "\n";
                        // cout << "maha" << Mahalanobis(kpp,kcc,cov_mat.inv()) <<endl;
                    }
                    catch (const std::exception &e)
                    {
                        Distance << "-1\n";
                        std::cerr << e.what() << '\n';
                    }
                }
            }
        }
    }
    if (save_to_file)
    {
        ofstream PP(folder + "/points_peak.txt");
        ofstream PM(folder + "/points_mean.txt");
        ofstream D(folder + "/distance.txt");

        PP << PP_string.str();
        PM << PM_string.str();
        D << Distance.str();

        // Close the file
        PP.close();
        PM.close();
        D.close();
    }
}
