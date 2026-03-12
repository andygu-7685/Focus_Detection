#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc.hpp> // Image processing (drawing, resizing)
#include <opencv2/highgui.hpp> // GUI (imshow, namedWindow)

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iomanip>

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





static Mat increase_contrast(const Mat &image, const string &method = "clahe") {
    if (image.empty()) return image;
    
    // 1. If it's already grayscale, proceed
    if (image.channels() == 1) {
        // 2. Upload CPU Mat to GPU UMat
        // This is where the data moves to the Intel Graphics memory
        UMat u_gray = image.getUMat(ACCESS_READ);
        UMat u_dst;
        Ptr<CLAHE> local_clahe = createCLAHE(4.0, Size(8,8));

        //cvtimer.start();
        
        // 3. This execution now happens on the GPU
        local_clahe->apply(u_gray, u_dst); 
        
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
    //cout << "Lightest side average: " << lightest_side_avg << "\n";
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




// static Mat threshold_to_black(const Mat &img, int thresh = 150) {
//     if (img.empty()) return img;
//     Mat out = img.clone();
//     if (out.channels() != 3) {
//         // single channel
//         for (int y = 0; y < out.rows; ++y) {
//             uchar* p = out.ptr<uchar>(y);
//             for (int x = 0; x < out.cols; ++x) if (p[x] < thresh) p[x] = 0;
//         }
//     }
//     return out;
// }



static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " <folder> [--out path] [--block-size N] [--threshold T]\n";
}



pair<double, Mat> focus_score(const Mat& img, int block_size = 6, int threshold_val = 180, const string& out_path = "") {

    // cvtimer.reset();
    // cvtimer.start();

    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else if (img.channels() == 4) cvtColor(img, gray, COLOR_BGRA2GRAY);
    else gray = img;
    
    Mat out_img = increase_contrast(gray, "clahe");
    //imwrite(out_path + "_clahe.png", out_img);
    out_img = block_average_gray(out_img, block_size);
    //imwrite(out_path + "_block_average.png", out_img);
    out_img = threshold_to_black(out_img, threshold_val);
    //imwrite(out_path + "_thresholded.png", out_img);

    int non_black_pixel = count_non_black_pixels(out_img);
    double pct = non_black_pixel * 100.0 / 3000.0;
    cout.setf(std::ios::fixed); cout.precision(2);

    // cvtimer.stop();

    //cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;


    return make_pair(pct, out_img);
}



// Struct to hold our ranking data
struct ImageScore {
    string filename;
    double score;
};

// producer/consumer queue and synchronization
queue<pair<string, cv::Mat>> imageQueue;
mutex queue_mtx;
condition_variable queue_cv;
bool done_loading = false;

vector<ImageScore> rankings;
mutex rankings_mtx;

// producer thread reads files recursively and pushes them into the queue
void producer(const string& input_folder) {
    for (const auto& entry : fs::recursive_directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                string path = entry.path().string();
                string filename = entry.path().filename().string();
                Mat img = imread(path, IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                {
                    lock_guard<mutex> lk(queue_mtx);
                    imageQueue.push({filename, img});
                }
                queue_cv.notify_one();
            }
        }
    }

    // signal consumers that loading is finished
    {
        lock_guard<mutex> lk(queue_mtx);
        done_loading = true;
    }
    queue_cv.notify_all();
}

// consumer threads pull images off the queue and process them
void consumer(const string& output_folder, int block_size, int threshold_val) {
    while (true) {
        pair<string, cv::Mat> image_pair;
        {
            unique_lock<mutex> lk(queue_mtx);
            queue_cv.wait(lk, []{ return !imageQueue.empty() || done_loading; });
            if (imageQueue.empty() && done_loading)         //exit only when there's signal and queue is empty
                break;
            image_pair = imageQueue.front();
            imageQueue.pop();
        }

        const string& filename = image_pair.first;
        Mat img = image_pair.second;
        string out_path = output_folder + "/" + filename.substr(0, filename.find_last_of("."));

        cvtimer.reset(); 
        cvtimer.start();
        
        auto result = focus_score(img, block_size, threshold_val, out_path);
        double score = result.first;
        Mat processed = result.second;
        cvtimer.stop();
        std::cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;

        {
            lock_guard<mutex> lk(rankings_mtx);
            rankings.push_back({filename, score});
        }

    }
}



int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }


    string input_folder = argv[1];
    string output_folder = "output_images"; // Default output folder
    int block_size = 6;
    int threshold_val = 50;

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

    // start producer and a pool of consumer threads
    thread prod(producer, input_folder);
    unsigned int nworkers = max(1u, thread::hardware_concurrency());
    vector<thread> workers;
    for (unsigned int i = 0; i < nworkers; ++i)
        workers.emplace_back(consumer, output_folder, block_size, threshold_val);

    prod.join();
    for (auto &t : workers) t.join();

    // after all processing, sort and display rankings
    sort(rankings.begin(), rankings.end(), [](const ImageScore& a, const ImageScore& b) {
        return a.score > b.score;
    });

    if (!rankings.empty()) {
        cout << "\n--- Ranked Images (Highest Score First) ---\n";
        cout << fixed << setprecision(2);
        for (const auto& item : rankings) {
            cout << item.filename << " : " << item.score << "%" << endl;
        }
    }

    return 0;
}