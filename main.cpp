#include <iostream>
#include <bitset>
#include <opencv2/opencv.hpp>

#define OPTIONAL_PROC
#define MEASURE_MARGIN  20
#define IMG_PATH "D://Workspace//Qt//SimpleBrailleSegmetation//braille_sample2.png"

using namespace std;
using namespace cv;

typedef struct _braille{
    _braille() : rect(), value(0), index(0){}
    virtual ~_braille(){}
    Rect rect;
    int value;  // decimal
    int index;
}braille;

int main()
{
    // 1) read input image
    Mat inputImg;
    inputImg = imread(IMG_PATH, IMREAD_GRAYSCALE);
    imshow("(1)inputImg", inputImg);

    // 2) pre-processing input image
    // apply adaptive threshold
    Mat thresholdImg;
    adaptiveThreshold(inputImg, thresholdImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 10);

    //apply gaussian blur & erode and then threshold again
    Mat morphologyElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    GaussianBlur(thresholdImg, thresholdImg, Size(3, 3), 0);
    erode(thresholdImg, thresholdImg, morphologyElement3x3);
    threshold(thresholdImg, thresholdImg, 21, 255, THRESH_BINARY);
    imshow("(1.5)thresholdImg", thresholdImg);

    // make margin and resize from original image
    Mat measureImg = Mat(Size(thresholdImg.cols + MEASURE_MARGIN, thresholdImg.rows + MEASURE_MARGIN), CV_8UC1, 255);
    thresholdImg.copyTo(measureImg(Rect(MEASURE_MARGIN/2, MEASURE_MARGIN/2 ,thresholdImg.cols, thresholdImg.rows)));
    resize(measureImg, measureImg, measureImg.size()*2);
    imshow("(2)measureImg", measureImg);

    // 3) detect blobs
    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 2.0f * 2.0f;
    params.maxArea = 20.0f * 20.0f;

    Ptr<SimpleBlobDetector> blobDetector = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    blobDetector->detect(measureImg, keypoints);

    // check keypoints existance & draw keypoints for displaying
    if(keypoints.empty()){
        cout << "there is no braille existance condition" << endl;
        return 1;
    }
    Mat detectedImg = Mat(measureImg.size(), CV_8UC3);
    drawKeypoints(measureImg, keypoints, detectedImg, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("(3)detectedImg", detectedImg);


    // 4) normalize keypoints to coordinate line set
    float blobSize = 0.0f;
    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
        blobSize += keypoints[i].size;
    }
    blobSize /= keypoints.size();
    cout << "mean of the blob sizes : " << blobSize << endl;
    vector<int> coordinateX;
    vector<int> coordinateY;
    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
        bool isNew = true;
        for (vector<int>::iterator iter = coordinateX.begin(); iter < coordinateX.end(); ++iter) {
            if(abs(*iter - keypoints[i].pt.x) < blobSize){
                isNew = false;
                break;
            }
        }
        if(isNew){
            coordinateX.push_back((int)keypoints[i].pt.x);
        }

        isNew = true;
        for (vector<int>::iterator iter = coordinateY.begin(); iter < coordinateY.end(); ++iter) {
            if(abs(*iter - keypoints[i].pt.y) < blobSize){
                isNew = false;
                break;
            }
        }
        if(isNew){
            coordinateY.push_back((int)keypoints[i].pt.y);
        }
    }
    sort(coordinateX.begin(), coordinateX.end());
    sort(coordinateY.begin(), coordinateY.end());

#ifdef OPTIONAL_PROC
    // draw coordinate lines for displaying
    Mat coordinateImg = detectedImg.clone();
    for (int i = 0; i < static_cast<int>(coordinateX.size()); ++i) {
        line(coordinateImg, Point(coordinateX[i], 0), Point(coordinateX[i], coordinateImg.rows), Scalar(255, 0, 0));
    }
    for (int i = 0; i < static_cast<int>(coordinateY.size()); ++i) {
        line(coordinateImg, Point(0, coordinateY[i]), Point(coordinateImg.cols, coordinateY[i]), Scalar(255, 0, 0));
    }
    imshow("(4)coordinateImg", coordinateImg);
#endif


    // 5) move keypoints to the nearest coordinate point
    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
        int ditanceX = detectedImg.cols / 2;
        int ditanceY = detectedImg.rows / 2;
        int tempX = 0;
        int tempY = 0;
        for (int j = 0; j < static_cast<int>(coordinateX.size()); ++j) {
            if(ditanceX > abs(keypoints[i].pt.x - coordinateX[j])){
                ditanceX = abs(keypoints[i].pt.x - coordinateX[j]);
                tempX = coordinateX[j];
            }
        }
        keypoints[i].pt.x = tempX;

        for (int j = 0; j < static_cast<int>(coordinateY.size()); ++j) {
            if(ditanceY > abs(keypoints[i].pt.y - coordinateY[j])){
                ditanceY = abs(keypoints[i].pt.y - coordinateY[j]);
                tempY = coordinateY[j];
            }
        }
        keypoints[i].pt.y = tempY;
    }

    // make image from the edited keypoint set(draw line for display)
    Mat editedImg = Mat(detectedImg.size(), CV_8UC1);
    editedImg.setTo(255);
    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
        circle(editedImg, Point(keypoints[i].pt.x, keypoints[i].pt.y), blobSize / 2, Scalar(0), -1, LINE_AA);
    }
    imshow("(5)editedImg", editedImg);

#ifdef OPTIONAL_PROC
    Mat editedwithLineImg = editedImg.clone();
    for (int i = 0; i < static_cast<int>(coordinateX.size()); ++i) {
        line(editedwithLineImg, Point(coordinateX[i], 0), Point(coordinateX[i], editedwithLineImg.rows), Scalar(0));
    }
    for (int i = 0; i < static_cast<int>(coordinateY.size()); ++i) {
        line(editedwithLineImg, Point(0, coordinateY[i]), Point(editedwithLineImg.cols, coordinateY[i]), Scalar(0));
    }
    imshow("(5.5)editedwithLineImg", editedwithLineImg);
#endif


    // 6) segmentation braille rectangle
    int startXPos = 0;
    int index = 0;
    vector<braille> brailleSet;
    Mat segmentationImg = Mat(editedImg.size(), CV_8UC3);
    cvtColor(editedImg, segmentationImg, COLOR_GRAY2BGR);
    if((coordinateX[1] - coordinateX[0]) > (coordinateX[2] - coordinateX[1])){
        startXPos = 1;
    }
    for(int i = 0; i < static_cast<int>(coordinateY.size()) - 2; i += 3){
        for(int j = startXPos; j < static_cast<int>(coordinateX.size()) - 1; j += 2){
            braille tempBraille;
            Rect rect = Rect(Point(coordinateX[j] - blobSize / 2, coordinateY[i] - blobSize / 2),
                             Point(coordinateX[j + 1] + blobSize / 2, coordinateY[i + 2] + blobSize / 2));
            int value = 0;
            rectangle(segmentationImg, rect, Scalar(0, 0, 255));

            // set the braille value(2x3 matrix)
            for(int k = 0; k < 2; ++k){
                for(int l = 0; l < 3; ++l){
                    if(editedImg.at<uchar>(Point((int)coordinateX[j + k] , (int)coordinateY[i + l])) == 0){
                        value++;
                    }
                    value = value << 1;
                }
            }
            value = value >> 1;
            tempBraille.rect = rect;
            tempBraille.index = index++;
            tempBraille.value = value;
            brailleSet.push_back(tempBraille);
        }
    }
    if(brailleSet.empty()){
        cout << "there is no braille set !!" << endl;
        return 1;
    }
    imshow("(6)segmentationImg", segmentationImg);

#ifdef OPTIONAL_PROC
    // image of comparision with the input image
    Mat compareImg, resizedImg;
    cvtColor(inputImg, compareImg, COLOR_GRAY2BGR);
    resize(segmentationImg, resizedImg, segmentationImg.size() / 2);
    addWeighted(compareImg, 0.8, resizedImg(Rect(MEASURE_MARGIN/2, MEASURE_MARGIN/2 ,inputImg.cols, inputImg.rows)), 0.2, 0.0,  compareImg);
    imshow("(6.5)compareImg", compareImg);
#endif


    // 7) make result image
    Mat resultImg = Mat(Size(segmentationImg.size()), CV_8UC3);
    resultImg.setTo(255);
    addWeighted(resultImg, 0.8, segmentationImg, 0.2, 0.0,  resultImg);

    int intFontFace = FONT_HERSHEY_SIMPLEX;
    double dblFontScale = brailleSet[0].rect.size().width / 60.0;
    int intFontThickness = (int)std::round(dblFontScale * 2);

    for(int i = 0; i < static_cast<int>(brailleSet.size()); ++i){
        Point center, bottomLeft;
        center = (brailleSet[i].rect.tl() + brailleSet[i].rect.br()) / 2;
        center.x -= getTextSize(to_string(brailleSet[i].value), intFontFace, dblFontScale, intFontThickness, 0).width / 2;
        center.y += getTextSize(to_string(brailleSet[i].value), intFontFace, dblFontScale, intFontThickness, 0).height / 2;

        bottomLeft = Point(brailleSet[i].rect.tl().x, brailleSet[i].rect.br().y);
        bottomLeft.x -= blobSize / 2;
        bottomLeft.y += getTextSize(bitset<6>(brailleSet[i].value).to_string(), intFontFace, dblFontScale * 0.7, intFontThickness * 0.7, 0).height / 2 + blobSize / 2;

        putText(resultImg, to_string(brailleSet[i].value), center, intFontFace, dblFontScale, Scalar(255,0,0), intFontThickness);
        putText(resultImg, bitset<6>(brailleSet[i].value).to_string(), bottomLeft, intFontFace, dblFontScale * 0.7, Scalar(0,0,0), intFontThickness * 0.7);
    }
    imshow("(7)resultImg", resultImg);

    waitKey(0);
    return 0;
}
