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



static Mat convolve_stride(const Mat &img, const Mat &kernel, int jmp_size = 4) {
    // Output size will be roughly half
    int kernel_size = kernel.rows;
    Mat out(img.rows / jmp_size, img.cols / jmp_size, img.type());

    for (int y = 0; y < out.rows; y++) {
        for (int x = 0; x < out.cols; x++) {
            // Map output (x,y) to input (x*2, y*2)
            Rect roi(x * jmp_size, y * jmp_size, kernel_size, kernel_size);
            
            // Boundary check: ensure the 6x6 neighborhood fits
            if (roi.x + kernel_size <= img.cols && roi.y + kernel_size <= img.rows) {
                Mat neighborhood = img(roi);
                // Multiply neighborhood by kernel and sum
                // (Using dot product for efficiency)
                Mat neighborhood_float;
                neighborhood.convertTo(neighborhood_float, CV_32F);
                float sum = neighborhood_float.dot(kernel);
                out.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            }
        }
    }
    return out;
}

Mat getLaplacianOfGaussian13x13() {
    int size = 13;
    // Sigma controls the width of the positive peak. 
    // For an 11x11 spot, a sigma of ~2.0 is ideal.
    double sigma = 1.5; 
    Mat kernel(size, size, CV_32F);
    
    float sum = 0;
    int center = size / 2;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            // LoG Formula: -(1/(pi*sigma^4)) * (1 - (x^2+y^2)/(2*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
            float r2 = dx*dx + dy*dy;
            float s2 = sigma*sigma;
            float val = (1.0f - r2 / (2.0f * s2)) * exp(-r2 / (2.0f * s2));
            kernel.at<float>(y, x) = val;
            sum += val;
        }
    }

    // Force the sum to 0 so the background stays black
    // We subtract the average error from every pixel
    float offset = sum / (size * size);
    kernel -= offset;

    return kernel;
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

    if(true) {
        Mat blurred;
        GaussianBlur(out_img, blurred, Size(5, 5), 0);
        // 1. Create a simple 6x6 averaging kernel
        Mat kernel = getLaplacianOfGaussian13x13();
        out_img = convolve_stride(blurred, kernel);
    } else {
        out_img = block_average_gray(out_img, block_size);
        out_img = threshold_to_black(out_img, threshold_val);
    }

    //int non_black_pixel = count_non_black_pixels(out_img);
    double pct = 1; //non_black_pixel * 100.0 / 3000.0;
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

int main(int argc, char** argv) {
    //if (argc < 2) { print_usage(argv[0]); return 1; }
    cv::setUseOptimized(false);

    string input_folder = "C:\\Users\\USER\\Documents\\Research\\Focus_Algo\\Stack2"; //argv[1];
    string output_folder = "output_images"; // Default output folder
    int block_size = 13;
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