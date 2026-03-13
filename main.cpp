#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc.hpp> // Image processing (drawing, resizing)
#include <opencv2/highgui.hpp> // GUI (imshow, namedWindow)
#include "./includes/img_proc.h"   // Custom image processing functions

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <regex>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;
cv::TickMeter cvtimer;
std::ofstream logFile;










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



struct ImageScore {
    string filename;
    string folder;
    double voltage;
    double score;
    double time_ms;
    int quarter;
    int x_type;
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
                    imageQueue.push({path, img});
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

    // precompile some regexes used for extracting properties
    std::regex file_re(R"(diag_(X(?:\+1|\-1)?)_V([\d\.]+))");
    std::regex quarter_re(R"(Q([1-4]))");

    while (true) {
        pair<string, cv::Mat> image_pair;
        cvtimer.reset(); 
        cvtimer.start();
        {
            unique_lock<mutex> lk(queue_mtx);
            queue_cv.wait(lk, []{ return !imageQueue.empty() || done_loading; });
            if (imageQueue.empty() && done_loading)         //exit only when there's signal and queue is empty
                break;
            image_pair = imageQueue.front();
            imageQueue.pop();
        }

        Mat img = image_pair.second;
        string full_path = image_pair.first;

        size_t last_slash = full_path.find_last_of("/\\"); 
        string filename = (last_slash == string::npos) ? full_path : full_path.substr(last_slash + 1);
        string input_folder = full_path;

        size_t last_dot = filename.find_last_of(".");
        string base_name = (last_dot == string::npos) ? filename : filename.substr(0, last_dot);
        string out_path = output_folder + "/" + base_name;



        
        auto result = focus_score(img, block_size, threshold_val, out_path);
        double score = result.first;
        Mat processed = result.second;

        cvtimer.stop();
        double time_ms = cvtimer.getTimeMilli();
        std::cout << "Time in milli: " << time_ms << endl;



        // extract properties from filename
        double voltage = 0.0;
        int x_type = 0; // 0 = X, 1 = X+1, -1 = X-1
        std::smatch fm;
        if (std::regex_search(filename, fm, file_re)) {
            string xstr = fm[1];
            voltage = std::stod(fm[2]);
            if (xstr == "X+1") x_type = 1;
            else if (xstr == "X-1") x_type = -1;
        }

        // extract quarter from folder path
        int quarter = -1;
        std::smatch qm;
        if (std::regex_search(input_folder, qm, quarter_re)) {
            quarter = std::stoi(qm[1]);
        }



        {
            lock_guard<mutex> lk(rankings_mtx);
            rankings.push_back({filename, input_folder, voltage, result.first, time_ms, quarter, x_type});
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

    // Open log file for writing
    logFile.open("log_file.txt", std::ios::app);

    // after all processing, sort and display rankings
    sort(rankings.begin(), rankings.end(), [](const ImageScore& a, const ImageScore& b) {
        return a.score > b.score;
    });

    if (!rankings.empty()) {
        cout << "\n--- Ranked Images (Highest Score First) ---\n";
        cout << fixed << setprecision(2);
        if (logFile.is_open()) logFile << fixed << setprecision(2);
        for (const auto& item : rankings) {
            cout << item.filename << " : " << item.score << "%"
             << "  V=" << item.voltage
             << "  x=" << item.x_type
             << "  Q=" << item.quarter
             << "  time=" << item.time_ms << "ms"
             << "  folder=" << item.folder
             << endl;
            if (logFile.is_open()) {
                logFile << item.filename << " : " << item.score << "%"
                 << "  V=" << item.voltage
                 << "  x=" << item.x_type
                 << "  Q=" << item.quarter
                 << "  time=" << item.time_ms << "ms"
                 << "  folder=" << item.folder
                 << endl;
            }
        }
    }

    if (logFile.is_open()) logFile.close();
    return 0;
}