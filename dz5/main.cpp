#include <iostream>
#include <vector>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

template<class T>
void getSubsequencesOfLength(vector<vector<T>> &result,
                             const vector<T> &vec,
                             int length,
                             int idx = 0,
                             vector<T> subsequenceBuilder = {}) {

    if (subsequenceBuilder.size() == length) {
        result.emplace_back(subsequenceBuilder);
        return;
    }
    if (idx >= vec.size()) {
        return;
    }

    getSubsequencesOfLength(result, vec, length, idx + 1, subsequenceBuilder);
    subsequenceBuilder.emplace_back(vec[idx]);
    getSubsequencesOfLength(result, vec, length, idx + 1, subsequenceBuilder);
}

Scalar randomColor() {
    static RNG rng(12345);
    return Scalar(
            rng.uniform(0, 256),
            rng.uniform(0, 256),
            rng.uniform(0, 256)
    );
}

string findText(const Mat &frame, vector<Point> contour) {
    return "опционально";
}

int main() {
    Mat image, src, src_gray, grad;
    image = imread("test_marker.jpg", IMREAD_COLOR);
    if (image.empty()) {
        return -1;
    }
    GaussianBlur(image, src, Size(5, 5), 1, 1, BORDER_DEFAULT);
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Canny(src_gray, grad, 30, 90);
    vector<vector<Point>> contours;
    findContours(grad, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> hull(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        convexHull(contours[i], hull[i]);
    }


    Mat drawing = Mat::zeros(grad.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(drawing, hull, (int) i, randomColor());
    }

    imshow("drawing", drawing);
    waitKey(0);

    vector<int> possibleRectsIndices;
    for (size_t i = 0; i < contours.size(); i++) {
        auto areaCnt = contourArea(contours[i]);
        auto areaHull = contourArea(hull[i]);
        if (abs(areaHull - areaCnt) / areaHull < 0.1 && areaHull > 2000) {
            contours[i] = hull[i];
            possibleRectsIndices.emplace_back(i);
        } else {
            contours[i] = {};
        }
    }

    Mat possibleRectsImg = Mat::zeros(grad.size(), CV_8UC3);
    for (int i : possibleRectsIndices) {
        drawContours(possibleRectsImg, hull, i, randomColor());
    }
    imshow("possible rects", possibleRectsImg);
    waitKey(0);

    vector<vector<Point>> rects;
    vector<Point> centers = {};
    for (int i : possibleRectsIndices) {
        auto possibleRect = hull[i];
        vector<vector<Point>> possibleRects4Points;
        getSubsequencesOfLength(possibleRects4Points, possibleRect, 4);
        for (auto &possibleRect4Points:possibleRects4Points) {
            auto p1 = possibleRect4Points[0];
            auto p2 = possibleRect4Points[1];
            auto p3 = possibleRect4Points[2];
            auto p4 = possibleRect4Points[3];
            auto center = (p1 + p2 + p3 + p4) / 4;
            bool alreadyFound = false;
            for (auto &c:centers) {
                auto delta = center - c;
                if (delta.x * delta.x + delta.y * delta.y < 100) {
                    alreadyFound = true;
                    break;
                }
            }
            if (alreadyFound) {
                break;
            }
            if (contourArea(possibleRect4Points) / contourArea(possibleRect) < 0.9) {
                continue;
            }
            auto line1 = p2 - p1;
            auto line2 = p3 - p2;
            auto line3 = p4 - p3;
            auto line4 = p1 - p4;
            auto cos1 = line1.ddot(line3) /
                        sqrt(line1.x * line1.x + line1.y * line1.y) /
                        sqrt(line3.x * line3.x + line3.y * line3.y);
            auto cos2 = line2.ddot(line4) /
                        sqrt(line2.x * line2.x + line2.y * line2.y) /
                        sqrt(line4.x * line4.x + line4.y * line4.y);

            if (abs(cos1) > 0.9 && abs(cos2) > 0.9) {
                vector<Point> rect = {p1, p2, p3, p4};
                rects.emplace_back(rect);
                centers.emplace_back(center);
                break;
            }
        }
    }

    Mat rectsImg = Mat::zeros(grad.size(), CV_8UC3);
    for (int i = 0; i < rects.size(); ++i) {
        drawContours(rectsImg, rects, i, Scalar(255, 0, 255));
    }

    imshow("rects", rectsImg);

    for (auto &rect : rects) {
        auto text = findText(image, rect);
        cout << text << endl;
    }

    waitKey(0);
    return 0;
}
