#include "./rs_utility.h"

// Convert the rs color frame into a openCV Mat
// BGR_RGB -> 1 do not convert to RGB
cv::Mat frame_to_mat(const rs2::frame &f, bool BGR_RGB = 0)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
        if (!BGR_RGB)
            cvtColor(r, r, COLOR_BGR2RGB);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

rs2::config config_pipe(bool flag_small)
{
    rs2::context ctx;
    std::cout << "hello from librealsense - " << RS2_API_VERSION_STR << std::endl;
    std::cout << "You have " << ctx.query_devices().size() << " RealSense devices connected" << std::endl;
    rs2::config cfg;
    if (flag_small)
    {
        cfg.enable_stream(RS2_STREAM_COLOR, -1, 640, 480, RS2_FORMAT_RGB8);
    }
    else
    {
        cfg.enable_stream(RS2_STREAM_COLOR, -1, 1280, 720, RS2_FORMAT_RGB8);
    }

    cfg.enable_stream(RS2_STREAM_DEPTH, -1, 1280, 720, RS2_FORMAT_Z16);
    return cfg;
}