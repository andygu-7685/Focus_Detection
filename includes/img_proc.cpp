#include "img_proc.h"

using namespace cv;
using namespace std;

int count_non_black_pixels(const Mat &img) {
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




Mat block_average_gray(const Mat &gray, int block_size) {
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





Mat increase_contrast(const Mat &image, const string &method) {
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
double get_lightest_side_average_strided(const cv::Mat& img, int row_stride) {
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



Mat threshold_to_black(const Mat &img, int thresh) {
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