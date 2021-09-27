#include "./lib/followme.h"
#include "./lib/people.h"

#include <fstream>

// #include "./lib/pose_extractor.h"

int main(int argc, char *argv[])
{
    cv::VideoCapture cap(argv[1]);
    // cv::VideoCapture cap("/home/andrea/Desktop/video_test_multiperson/%4d.jpg");
    //_____________________________________________________________________________
    cv::Mat img, img_full_size;
    std::vector<std::vector<Point>> lista_punti = std::vector<std::vector<Point>>(18);
    // load single image
    // Mat img = cv::imread("../group.jpg");

    cv::dnn::Net network_pose;
    load_net_pose(&network_pose);

    //________ timing ___________
    // Start and end times
    time_t start, end;
    // Start time
    // std::time(&start);
    double count_frame = 0;
    //___________________________

    //________ people tracking __
    TrackerPeople::PeopleList peopleTrack;
    //___________________________
    int tw = 10;
    for (;;)
    {
        // start timing
        if (count_frame == 1)
            std::time(&start);

        cout << "frame " << count_frame << endl;
        cap >> img_full_size;
        if (img_full_size.empty())
        {
            break;
        }
        img_full_size.copyTo(img);

        // people tracker
        peopleTrack.update(img);
        peopleTrack.plotBoxes(img);
        peopleTrack.plotTrack(img);
        peopleTrack.plotStoryPosition(img);

        // show the image
        cv::imshow("Output", img);

        // wait for show
        int k = waitKey(tw);
        if (k == 'q' | k == 27)
            break;
        else if (k == 'w')
        {
            tw = 0;
        }
        else if (k == 'r')
        {
            tw = 20;
        }

        count_frame++;
    }

    // stop timer for FPS evaluation
    std::time(&end);

    // Time elapsed
    double seconds = difftime(end, start);

    std::cout << "Time taken    : " << seconds << " seconds" << std::endl;
    // Calculate frames per second
    auto fps = count_frame / seconds;
    std::cout << "Estimated FPS : " << fps << std::endl;

    return 0;
}
