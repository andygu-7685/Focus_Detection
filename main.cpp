#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc.hpp> // Image processing (drawing, resizing)
#include <opencv2/highgui.hpp> // GUI (imshow, namedWindow)
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

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
    
//     Mat out = gray.clone();
//     int h = gray.rows;
//     int w = gray.cols;
//     int bs = std::max(1, block_size);

//     for (int y = 0; y < h; y += bs) {
//         // Calculate remaining height to avoid going out of bounds
//         int block_h = std::min(bs, h - y);
        
//         for (int x = 0; x < w; x += bs) {
//             int block_w = std::min(bs, w - x);
            
//             // Define the rectangle correctly
//             Rect r(x, y, block_w, block_h);
            
//             // Create the ROI (Region of Interest)
//             Mat block = gray(r).clone(); // This ensures the data starts at an aligned address
//             double sum = 0;
//             int count = 0;
//             for (int i = 0; i < block.rows; i++) {
//                 for (int j = 0; j < block.cols; j++) {
//                     sum += block.at<uchar>(i, j);
//                     count++;
//                 }
//             }
//             double meanVal = sum / count;
//             out(r).setTo((uchar)std::round(meanVal));
            
//             // Use the ROI of the output matrix to set the value
//             Mat out_roi = out(r);
//             out_roi.setTo(meanVal); 
//         }
//     }
//     return out;
// }



static Mat block_average_gray(const Mat &gray, int block_size = 6) {
    CV_Assert(gray.type() == CV_8UC1);
    int bs = std::max(1, block_size);

    // Calculate the size of the reduced image
    Size smallSize(gray.cols / bs, gray.rows / bs);
    if (smallSize.width == 0 || smallSize.height == 0) return gray.clone();

    // 1. Downscale: INTER_AREA is mathematically equivalent to your 'mean' loop
    Mat small;
    resize(gray, small, smallSize, 0, 0, INTER_AREA);

    // 2. Upscale: INTER_NEAREST stretches those averages back into blocks
    Mat out;
    resize(small, out, gray.size(), 0, 0, INTER_NEAREST);

    return out;
}




static Ptr<CLAHE> global_clahe = createCLAHE(4.0, Size(8,8));

static Mat increase_contrast(const Mat &image, const string &method = "clahe") {
    if (image.empty()) return image;
    if (image.channels() == 1) {
        Mat gray = image;
        // if (method == "hist") {
        //     Mat dst; equalizeHist(gray, dst); return dst;
        // }
        Mat dst;
        cvtimer.start();
        global_clahe->apply(gray, dst); 
        cvtimer.stop();
        return dst;
    }
    return image; 


    // // Color image (BGR)
    // if (method == "hist") {
    //     Mat ycrcb; cvtColor(image, ycrcb, COLOR_BGR2YCrCb);
    //     std::vector<Mat> ch; split(ycrcb, ch);
    //     Mat y_eq; equalizeHist(ch[0], y_eq);
    //     ch[0] = y_eq;
    //     Mat merged; merge(ch, merged);
    //     Mat out; cvtColor(merged, out, COLOR_YCrCb2BGR);
    //     return out;
    // }

    // Mat lab; cvtColor(image, lab, COLOR_BGR2Lab);
    // std::vector<Mat> labch; split(lab, labch);
    // Ptr<CLAHE> clahe = createCLAHE(clip_limit, tile_grid_size);
    // Mat l_clahe; clahe->apply(labch[0], l_clahe);
    // labch[0] = l_clahe;
    // Mat lab_clahe; merge(labch, lab_clahe);
    // Mat out; cvtColor(lab_clahe, out, COLOR_Lab2BGR);
    // return out;
}

static Mat threshold_to_black(const Mat &img, int thresh = 150) {
    if (img.empty()) return img;
    Mat out = img.clone();
    if (out.channels() == 3) {
        // Mat gray; cvtColor(out, gray, COLOR_BGR2GRAY);
        // for (int y = 0; y < out.rows; ++y) {
        //     const uchar* gp = gray.ptr<uchar>(y);
        //     Vec3b* bp = out.ptr<Vec3b>(y);
        //     for (int x = 0; x < out.cols; ++x) {
        //         if (gp[x] < thresh) bp[x] = Vec3b(0,0,0);
        //     }
        // }
        // return out;
    } else {
        // single channel
        for (int y = 0; y < out.rows; ++y) {
            uchar* p = out.ptr<uchar>(y);
            for (int x = 0; x < out.cols; ++x) if (p[x] < thresh) p[x] = 0;
        }
        return out;
    }
    return out;
}

static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " <folder> [--out path] [--block-size N] [--threshold T]\n";
}







pair<double, Mat> focus_score(const Mat& img, int block_size = 6, int threshold_val = 180) {

    cvtimer.reset();
    //cvtimer.start();

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

    //cvtimer.stop();

    cout << "Time in milli: " << cvtimer.getTimeMilli() << endl;
    return make_pair(pct, out_img);
}










// int main(int argc, char** argv) {
//     if (argc < 2) { print_usage(argv[0]); return 1; }
    
//     string image_path;
//     string out_path;
//     int block_size = 6;
//     int threshold_val = 180;
//     bool show = false;

//     image_path = argv[1];
//     for (int i = 2; i < argc; ++i) {
//         string a = argv[i];
//         if (a == "--out" && i + 1 < argc) { out_path = argv[++i]; }
//         else if (a == "--block-size" && i + 1 < argc) { block_size = stoi(argv[++i]); }
//         else if (a == "--threshold" && i + 1 < argc) { threshold_val = stoi(argv[++i]); }
//         else if (a == "--show") { show = true; }
//         else { cerr << "Unknown arg: " << a << "\n"; print_usage(argv[0]); return 1; }
//     }

//     Mat img = imread(image_path, IMREAD_UNCHANGED);
//     if (img.empty()) { cerr << "Error: could not read image: " << image_path << "\n"; return 2; }


//     //-----------------------------------------------------------------------------------------------------------

//     pair<double, Mat> result = focus_score(img, block_size, threshold_val);
//     double pct = result.first;
//     Mat out_img = result.second;

//     //-----------------------------------------------------------------------------------------------------------


//     cout << "Non-black pixels: " << pct << "%\n";
//     if (pct > 100) cout << "The Camera is focused\n";
//     else if (pct > 75) cout << "The Camera is likely focused\n";
//     else cout << "The Camera is likely not focused\n";

//     if (out_path.empty()) {
//         // create default output name
//         size_t pos = image_path.find_last_of("/\\");
//         string base = (pos==string::npos) ? image_path : image_path.substr(pos+1);
//         size_t dot = base.find_last_of('.');
//         if (dot != string::npos) base = base.substr(0, dot);
//         out_path = base + "_blockavg_" + to_string(block_size) + "x" + to_string(block_size) + ".png";
//     }

//     imwrite(out_path, out_img);
//     cout << "Wrote: " << out_path << "\n";

//     if (show) {
//         imshow("blockavg", out_img);
//         waitKey(0);
//         destroyAllWindows();
//     }

//     return 0;
// }




namespace fs = std::filesystem;

// Struct to hold our ranking data
struct ImageScore {
    string filename;
    double score;
};

int main(int argc, char** argv) {
    //if (argc < 2) { print_usage(argv[0]); return 1; }

    //string input_folder = argv[1];
    string input_folder = "C:\\Users\\USER\\Documents\\Research\\Focus_Algo\\Stack";
    string output_folder = "output_images"; // Default output folder
    int block_size = 6;
    int threshold_val = 180;

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

    vector<ImageScore> rankings;

    // Iterate through all files in the folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            string filename = entry.path().filename().string();

            Mat img = imread(path, IMREAD_UNCHANGED);
            if (img.empty()) continue; // Skip non-image files

            // Process the image
            pair<double, Mat> result = focus_score(img, block_size, threshold_val);
            
            // Store the score
            rankings.push_back({filename, result.first});

            // Save the processed image to the output folder
            string out_path = output_folder + "/" + entry.path().stem().string() + "_processed.png";
            imwrite(out_path, result.second);
        }
    }

    // Sort rankings from largest to smallest score
    sort(rankings.begin(), rankings.end(), [](const ImageScore& a, const ImageScore& b) {
        return a.score > b.score;
    });

    // Output the ranked list
    cout << "\n--- Ranked Images (Highest Score First) ---\n";
    cout << fixed << setprecision(2);
    for (const auto& item : rankings) {
        cout << item.filename << " : " << item.score << "%" << endl;
    }

    return 0;
}