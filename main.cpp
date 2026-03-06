#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc.hpp> // Image processing (drawing, resizing)
//#include <opencv2/highgui.hpp> // GUI (imshow, namedWindow)
#include <opencv2/imgcodecs.hpp> // Image file reading/writing (imread, imwrite)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;
namespace py = pybind11;
cv::TickMeter cvtimer;



static int count_non_black_pixels(const Mat &img) {
    if (img.empty()) return 0;
    CV_Assert(img.type() == CV_8UC1); // Ensure it's grayscale

    int count = 0;
    for (int i = 0; i < img.rows; i++) {
        const uchar* row = img.ptr<uchar>(i);
        for (int j = 0; j < img.cols; j++) {
            if (row[j] != 0) count++;
        }
    }
    return count;
}



// static Mat block_average_gray(const Mat &gray, int block_size = 6) {
//     CV_Assert(gray.type() == CV_8UC1);
//     int bs = std::max(1, block_size);

//     // Calculate the size of the reduced image
//     Size smallSize(gray.cols / bs, gray.rows / bs);
//     if (smallSize.width == 0 || smallSize.height == 0) return gray.clone();

//     // 1. Downscale: INTER_AREA is mathematically equivalent to your 'mean' loop
//     Mat small;
//     resize(gray, small, smallSize, 0, 0, INTER_AREA);

//     // 2. Upscale: INTER_NEAREST stretches those averages back into blocks
//     Mat out;
//     resize(small, out, gray.size(), 0, 0, INTER_NEAREST);

//     return out;
// }


static Mat block_average_gray(const Mat &gray, int block_size = 6) {
    CV_Assert(gray.type() == CV_8UC1);
    int bs = std::max(1, block_size);

    // Calculate the size of the reduced image
    Size smallSize(gray.cols / bs, gray.rows / bs);
    if (smallSize.width == 0 || smallSize.height == 0) return gray.clone();

    // 1. Move to UMat (GPU/OpenCL buffer)
    UMat u_gray = gray.getUMat(ACCESS_READ);
    UMat u_small, u_out;

    // 2. Downscale: Handled by OpenCL kernel
    resize(u_gray, u_small, smallSize, 0, 0, INTER_AREA);

    // 3. Upscale: Handled by OpenCL kernel
    resize(u_small, u_out, gray.size(), 0, 0, INTER_NEAREST);

    // 4. Return as Mat (implicitly moves data back to CPU RAM)
    return u_out.getMat(ACCESS_READ).clone();
}



static Ptr<CLAHE> global_clahe = createCLAHE(4.0, Size(8,8));

static Mat increase_contrast(const Mat &image, const string &method = "clahe") {
    if (image.empty()) return image;
    
    // 1. If it's already grayscale, proceed
    if (image.channels() == 1) {
        // 2. Upload CPU Mat to GPU UMat
        // This is where the data moves to the Intel Graphics memory
        UMat u_gray = image.getUMat(ACCESS_READ);
        UMat u_dst;

        //cvtimer.start();
        
        // 3. This execution now happens on the GPU
        global_clahe->apply(u_gray, u_dst); 
        
        //cvtimer.stop();

        // 4. Download result back to a CPU Mat to return it
        Mat dst;
        u_dst.copyTo(dst);
        return dst;
    }
    
    return image; 
}



static Mat threshold_to_black(const Mat &img, int thresh = 150) {
    if (img.empty()) return img;
    Mat out = img.clone();
    if (out.channels() != 3) {
        // single channel
        for (int y = 0; y < out.rows; ++y) {
            uchar* p = out.ptr<uchar>(y);
            for (int x = 0; x < out.cols; ++x) if (p[x] < thresh) p[x] = 0;
        }
    }
    return out;
}



static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " <folder> [--out path] [--block-size N] [--threshold T]\n";
}



pair<double, Mat> focus_score(const Mat& img, int block_size = 6, int threshold_val = 180) {

    cvtimer.reset();
    cvtimer.start();

    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else if (img.channels() == 4) cvtColor(img, gray, COLOR_BGRA2GRAY);
    else gray = img;
    
    Mat out_img = increase_contrast(gray, "clahe");
    out_img = block_average_gray(out_img, block_size);
    out_img = threshold_to_black(out_img, threshold_val);

    int non_black_pixel = count_non_black_pixels(out_img);
    double pct = non_black_pixel * 100.0 / 3000.0;
    cout.setf(std::ios::fixed); cout.precision(2);

    cvtimer.stop();

    cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;
    return make_pair(pct, out_img);
}









// Struct to hold our ranking data
struct ImageScore {
    string filename;
    double score;
};

vector<pair<string, double>> process_folder(
    string input_folder, 
    string output_folder = "output_images", 
    int block_size = 6, 
    int threshold_val = 180
) {
    // 1. Create output directory
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }

    vector<ImageScore> rankings;

    // 2. Iterate and Process
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            string filename = entry.path().filename().string();

            Mat img = imread(path, IMREAD_UNCHANGED);
            if (img.empty()) continue; 

            // Your focus_score logic
            pair<double, Mat> result = focus_score(img, block_size, threshold_val);
            
            rankings.push_back({filename, result.first});

            string out_path = output_folder + "/" + entry.path().stem().string() + "_processed.png";
            imwrite(out_path, result.second);
        }
    }

    // 3. Sort
    sort(rankings.begin(), rankings.end(), [](const ImageScore& a, const ImageScore& b) {
        return a.score > b.score;
    });

    // 4. Convert to a Python-friendly format (Vector of Pairs)
    vector<pair<string, double>> output;
    for (const auto& item : rankings) {
        output.push_back({item.filename, item.score});
    }

    return output;
}

//Pybind
PYBIND11_MODULE(focus_algo, m) {
    m.def("process_folder", &process_folder, 
          "Processes a folder of images and returns ranked scores",
          py::arg("input_folder"),
          py::arg("output_folder") = "output_images",
          py::arg("block_size") = 6,
          py::arg("threshold_val") = 180);
}