#ifndef IMG_PROC_H
#define IMG_PROC_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

// core image processing helpers used by the project

int count_non_black_pixels(const cv::Mat &img);

cv::Mat block_average_gray(const cv::Mat &gray, int block_size = 6);

cv::Mat increase_contrast(const cv::Mat &image, const std::string &method = "clahe");

double get_lightest_side_average_strided(const cv::Mat& img, int row_stride = 5);

cv::Mat threshold_to_black(const cv::Mat &img, int thresh = 150);

#endif // IMG_PROC_H
