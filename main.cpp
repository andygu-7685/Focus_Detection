#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc.hpp> // Image processing (drawing, resizing)
#include <opencv2/highgui.hpp> // GUI (imshow, namedWindow)
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;
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


// avoided AVX HAL for this function since it causes crash on intel graphics
double get_lightest_side_average_strided(const cv::Mat& img, int row_stride = 5) {
    if (img.empty()) return 0.0;

    cv::Mat gray;
    if (img.channels() > 1) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;

    int mid = gray.cols / 2;
    long long left_sum = 0;
    long long right_sum = 0;
    long long left_count = 0;
    long long right_count = 0;

    // Process row by row using raw pointers (bypasses AVX HAL)
    for (int y = 0; y < gray.rows; y += row_stride) {
        const uchar* ptr = gray.ptr<uchar>(y);
        
        // Manual sum for left side
        for (int x = 0; x < mid; ++x) {
            left_sum += ptr[x];
            left_count++;
        }

        // Manual sum for right side
        for (int x = mid; x < gray.cols; ++x) {
            right_sum += ptr[x];
            right_count++;
        }
    }

    if (left_count == 0 || right_count == 0) return 0.0;

    double left_avg = (double)left_sum / left_count;
    double right_avg = (double)right_sum / right_count;

    return std::max(left_avg, right_avg);
}



static Mat threshold_to_black(const Mat &img, int thresh = 150) {
    if (img.empty()) return img;
    Mat out = img.clone();
    double lightest_side_avg = get_lightest_side_average_strided(out);
    cout << "Lightest side average: " << lightest_side_avg << "\n";
    if (out.channels() != 3) {
        // single channel
        for (int y = 0; y < out.rows; ++y) {
            uchar* p = out.ptr<uchar>(y);
            for (int x = 0; x < out.cols; ++x) 
            {
                if (p[x] - lightest_side_avg < thresh) p[x] = 0;
            }
        }
    }
    return out;
}





static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " <folder> [--out path] [--block-size N] [--threshold T]\n";
}



pair<double, Mat> focus_score(const Mat& img, int block_size = 6, int threshold_val = 180, const string& out_path = "") {

    // cvtimer.reset();
    // cvtimer.start();
    // cv::ocl::setUseOpenCL(false);
    // cv::setUseOptimized(false);

    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else if (img.channels() == 4) cvtColor(img, gray, COLOR_BGRA2GRAY);
    else gray = img;
    
    Mat clahe_img = increase_contrast(gray, "clahe");
    Mat block_img = block_average_gray(clahe_img, block_size);
    // Mat block_img;
    // block_size = (block_size % 2 == 0) ? block_size + 1 : block_size;
    // GaussianBlur(clahe_img, block_img, Size(block_size, block_size), 0);
    Mat out_img = threshold_to_black(block_img, threshold_val);

    int non_black_pixel = count_non_black_pixels(out_img);
    double pct = non_black_pixel * 100.0 / 3000.0;
    cout.setf(std::ios::fixed); cout.precision(2);

    // cvtimer.stop();

    //cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;

    // Save the processed image to the output folder

    imwrite(out_path + "_clahe.png", clahe_img);
    imwrite(out_path + "_block.png", block_img);
    imwrite(out_path + "_processed.png", out_img);

    return make_pair(pct, out_img);
}



// Struct to hold our ranking data
struct ImageScore {
    string filename;
    double score;
};


void batch_focus_evaluation(string input_folder, string output_folder, int block_size = 6, int threshold_val = 180){
    cout << "\nCurrent Folder: " << input_folder << "\n";

    vector<ImageScore> rankings;
    // Iterate through all files in the folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file() && (entry.path().extension().string() == ".png" ||
                                       entry.path().extension().string() == ".jpg" ||
                                       entry.path().extension().string() == ".jpeg")) {
            string path = entry.path().string();
            string filename = entry.path().filename().string();

            cvtimer.reset();
            cvtimer.start();

            Mat img = imread(path, IMREAD_GRAYSCALE);
            if (img.empty()) continue; // Skip non-image files
            string out_path = output_folder + "/" + entry.path().stem().string();

            // Process the image
            pair<double, Mat> result = focus_score(img, block_size, threshold_val, out_path);
            
            // Store the score
            rankings.push_back({filename, result.first});


            cvtimer.stop();
            cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;
        }
        else if(entry.is_directory()) {
            batch_focus_evaluation(entry.path().string(), output_folder, block_size, threshold_val);
        }
    }

    // Sort rankings from largest to smallest score
    sort(rankings.begin(), rankings.end(), [](const ImageScore& a, const ImageScore& b) {
        return a.score > b.score;
    });

    // Output the ranked list
    if(rankings.empty()) return;
    cout << "\n--- Ranked Images (Highest Score First) ---\n";
    cout << fixed << setprecision(2);
    for (const auto& item : rankings) {
        cout << item.filename << " : " << item.score << "%" << endl;
    }
}



int main(int argc, char** argv) {
    //if (argc < 2) { print_usage(argv[0]); return 1; }

    //string input_folder = argv[1];
    string input_folder = "C:\\Users\\USER\\Documents\\Research\\Focus_Algo\\Stack";
    string output_folder = "output_images"; // Default output folder
    int block_size = 6;
    int threshold_val = 50;            //old: 180

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--out" && i + 1 < argc) { output_folder = argv[++i]; }
        else if (a == "--block-size" && i + 1 < argc) { block_size = stoi(argv[++i]); }
        else if (a == "--threshold" && i + 1 < argc) { threshold_val = stoi(argv[++i]); }
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }

    batch_focus_evaluation(input_folder, output_folder, block_size, threshold_val);


    return 0;
}