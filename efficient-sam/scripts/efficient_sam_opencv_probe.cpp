#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Copyright (c) 2026 OpenShot Studios, LLC
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Prompt {
    float x = 0.0f;
    float y = 0.0f;
    float label = 1.0f;
};

void printShape(const std::string& name, const cv::Mat& mat)
{
    std::cout << name << ":";
    for (int i = 0; i < mat.dims; ++i)
        std::cout << (i == 0 ? " [" : "x") << mat.size[i];
    std::cout << "]" << std::endl;
}

Prompt parsePrompt(const std::string& value)
{
    std::stringstream stream(value);
    std::string item;
    std::vector<float> parts;
    while (std::getline(stream, item, ',')) {
        parts.push_back(std::stof(item));
    }
    if (parts.size() != 3)
        throw std::runtime_error("prompt must be x,y,label");
    return {parts[0], parts[1], parts[2]};
}

cv::Mat imageBlob(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(1024, 1024), 0.0, 0.0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    return cv::dnn::blobFromImage(rgb);
}

void makePromptBlobs(
    const std::vector<Prompt>& prompts,
    cv::Size originalSize,
    cv::Mat& pointBlob,
    cv::Mat& labelBlob,
    std::vector<cv::Point>& backgroundPoints)
{
    constexpr int maxPrompts = 6;
    std::vector<Prompt> modelPrompts;
    const float sx = 1024.0f / static_cast<float>(originalSize.width);
    const float sy = 1024.0f / static_cast<float>(originalSize.height);
    for (const Prompt& prompt : prompts) {
        if (prompt.label == -1.0f) {
            backgroundPoints.emplace_back(
                static_cast<int>(std::lround(prompt.x * sx)),
                static_cast<int>(std::lround(prompt.y * sy)));
        } else {
            modelPrompts.push_back(prompt);
        }
    }
    if (modelPrompts.empty())
        throw std::runtime_error("at least one foreground or box prompt is required");
    if (modelPrompts.size() > maxPrompts)
        throw std::runtime_error("EfficientSAM OpenCV model accepts at most six model prompts");

    const int pointShape[] = {1, 1, maxPrompts, 2};
    pointBlob = cv::Mat(4, pointShape, CV_32F, cv::Scalar(0.0f));
    const int labelShape[] = {1, 1, maxPrompts, 1};
    labelBlob = cv::Mat(4, labelShape, CV_32F, cv::Scalar(-1.0f));

    float* points = pointBlob.ptr<float>();
    float* labels = labelBlob.ptr<float>();
    for (size_t i = 0; i < modelPrompts.size(); ++i) {
        points[i * 2] = modelPrompts[i].x * sx;
        points[i * 2 + 1] = modelPrompts[i].y * sy;
        labels[i] = modelPrompts[i].label;
    }
}

cv::Mat bestMask(const cv::Mat& outputMasks, const cv::Mat& iouPredictions, const std::vector<cv::Point>& backgroundPoints)
{
    if (outputMasks.dims != 5 || iouPredictions.dims != 3)
        throw std::runtime_error("unexpected EfficientSAM output rank");

    const int candidateCount = outputMasks.size[2];
    const int maskHeight = outputMasks.size[3];
    const int maskWidth = outputMasks.size[4];
    const float* ious = iouPredictions.ptr<float>();

    std::vector<int> order(candidateCount);
    for (int i = 0; i < candidateCount; ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return ious[a] > ious[b];
    });

    const float* masks = outputMasks.ptr<float>();
    const size_t candidatePixels = static_cast<size_t>(maskHeight) * static_cast<size_t>(maskWidth);
    cv::Mat fallback;
    for (int candidate : order) {
        cv::Mat mask(maskHeight, maskWidth, CV_8U, cv::Scalar(0));
        const float* src = masks + static_cast<size_t>(candidate) * candidatePixels;
        for (int y = 0; y < maskHeight; ++y) {
            uint8_t* row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < maskWidth; ++x)
                row[x] = src[y * maskWidth + x] >= 0.0f ? 255 : 0;
        }
        if (fallback.empty())
            fallback = mask.clone();

        bool containsBackground = false;
        for (const cv::Point& point : backgroundPoints) {
            int x = std::clamp(point.x, 0, maskWidth - 1);
            int y = std::clamp(point.y, 0, maskHeight - 1);
            if (mask.at<uint8_t>(y, x) != 0) {
                containsBackground = true;
                break;
            }
        }
        if (!containsBackground) {
            std::cout << "selected_candidate=" << candidate << " predicted_iou=" << ious[candidate] << std::endl;
            return mask;
        }
    }

    std::cout << "selected_candidate=" << order[0] << " predicted_iou=" << ious[order[0]]
              << " background_filter=fallback" << std::endl;
    return fallback;
}

cv::Mat overlayMask(const cv::Mat& bgr, const cv::Mat& mask)
{
    cv::Mat overlay = bgr.clone();
    for (int y = 0; y < overlay.rows; ++y) {
        const uint8_t* maskRow = mask.ptr<uint8_t>(y);
        cv::Vec3b* row = overlay.ptr<cv::Vec3b>(y);
        for (int x = 0; x < overlay.cols; ++x) {
            if (maskRow[x] == 0)
                continue;
            row[x][1] = static_cast<uint8_t>(std::min(255, static_cast<int>(row[x][1] * 0.35f + 180.0f)));
            row[x][2] = static_cast<uint8_t>(std::min(255, static_cast<int>(row[x][2] * 0.35f + 40.0f)));
        }
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(overlay, contours, -1, cv::Scalar(255, 255, 255), 2);
    return overlay;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " model.onnx image output_prefix x,y,label [x,y,label ...]\n"
                  << "labels: 1=foreground, -1=background filter, 2=box top-left, 3=box bottom-right\n";
        return 2;
    }

    try {
        const std::string modelPath = argv[1];
        const std::string imagePath = argv[2];
        const std::string outputPrefix = argv[3];

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
            throw std::runtime_error("could not read image: " + imagePath);

        std::vector<Prompt> prompts;
        for (int i = 4; i < argc; ++i)
            prompts.push_back(parsePrompt(argv[i]));

        cv::Mat points;
        cv::Mat labels;
        std::vector<cv::Point> backgroundPoints;
        makePromptBlobs(prompts, image.size(), points, labels, backgroundPoints);

        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
        net.setInput(imageBlob(image), "batched_images");
        net.setInput(points, "batched_point_coords");
        net.setInput(labels, "batched_point_labels");

        std::vector<cv::Mat> outputs;
        std::vector<cv::String> outputNames = {"output_masks", "iou_predictions"};
        net.forward(outputs, outputNames);
        printShape("output_masks", outputs[0]);
        printShape("iou_predictions", outputs[1]);

        cv::Mat mask1024 = bestMask(outputs[0], outputs[1], backgroundPoints);
        cv::Mat mask;
        cv::resize(mask1024, mask, image.size(), 0.0, 0.0, cv::INTER_NEAREST);

        const int pixels = cv::countNonZero(mask);
        std::cout << "mask_pixels=" << pixels << " image_pixels=" << image.total() << std::endl;

        if (!cv::imwrite(outputPrefix + "_mask.png", mask))
            throw std::runtime_error("failed writing mask output");
        if (!cv::imwrite(outputPrefix + "_overlay.png", overlayMask(image, mask)))
            throw std::runtime_error("failed writing overlay output");
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error:\n" << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "EfficientSAM OpenCV DNN forward: ok" << std::endl;
    return 0;
}
