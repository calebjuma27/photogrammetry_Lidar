#-------------------------------------------------------------------------------
# Name:        image_matching.cpp
# Purpose:
#
# Author:      caleb +AI
#
# Created:     19/03/2026
# Copyright:   (c) caleb 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <algorithm>

int main() {
    // 1. Load your images as grayscale
    cv::Mat img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        return -1;
    }

    // 2. Find keypoints and descriptors (ORB)
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

    // 3. Match features using Brute-Force (Hamming distance for ORB)
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    // Sort matches by distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    // 4. Extract location of matches
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    // 5. Find Fundamental Matrix and Rectify
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS);
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(pts1, pts2, F, img1.size(), H1, H2);

    cv::Mat img1_rect, img2_rect;
    cv::warpPerspective(img1, img1_rect, H1, img1.size());
    cv::warpPerspective(img2, img2_rect, H2, img1.size());

    // 6. Compute Dense Matching (Disparity Map)
    int numDisparities = 64; // Must be divisible by 16
    int blockSize = 5;

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, numDisparities, blockSize);
    stereo->setP1(8 * 3 * blockSize * blockSize);
    stereo->setP2(32 * 3 * blockSize * blockSize);
    stereo->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disparity;
    stereo->compute(img1_rect, img2_rect, disparity);

    // Normalize for visualization (SGBM outputs 16-bit fixed-point)
    cv::Mat disparity_vis;
    cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imshow("Dense Matching (Disparity)", disparity_vis);
    cv::waitKey(0);

    return 0;
}