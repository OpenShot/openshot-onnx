#include <iostream>
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static cv::Mat Blob(const std::vector<int>& shape, float value = 0.1f) {
    cv::Mat blob(static_cast<int>(shape.size()), shape.data(), CV_32F);
    blob.setTo(value);
    return blob;
}

static std::string SizeName(int width, int height) {
    if (width == height)
        return std::to_string(width);
    return std::to_string(width) + "x" + std::to_string(height);
}

static std::string ModelPath(const std::string& dir, const std::string& stem, int width, int height) {
    const std::string sizeName = SizeName(width, height);
    const std::string simplified = dir + "/" + stem + "-" + sizeName + "-sim.onnx";
    if (std::filesystem::exists(simplified))
        return simplified;
    return dir + "/" + stem + "-" + sizeName + ".onnx";
}

static std::string MemoryReadoutModelPath(const std::string& dir, int width, int height, int memoryFrames) {
    const std::string base = dir + "/cutie-memory-readout-floatmask-valid-" + SizeName(width, height)
        + "-m" + std::to_string(memoryFrames) + "-topk30";
    const std::string opencv = base + "-opencv.onnx";
    if (std::filesystem::exists(opencv))
        return opencv;
    const std::string simplified = base + "-sim.onnx";
    if (std::filesystem::exists(simplified))
        return simplified;
    return base + ".onnx";
}

static bool ForwardEncodeKey(const std::string& path, int width, int height) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
    net.setInput(Blob({1, 3, height, width}), "image");
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<cv::String>{"f16", "f8", "f4", "pix_feat", "key", "shrinkage", "selection"});
    std::cout << "encode_key outputs=" << outputs.size() << std::endl;
    return outputs.size() == 7;
}

static bool ForwardEncodeValue(const std::string& path, int width, int height) {
    const int stride16Width = width / 16;
    const int stride16Height = height / 16;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
    net.setInput(Blob({1, 3, height, width}), "image");
    net.setInput(Blob({1, 256, stride16Height, stride16Width}), "pix_feat");
    net.setInput(Blob({1, 1, 256, stride16Height, stride16Width}, 0.0f), "sensory");
    net.setInput(Blob({1, 1, height, width}), "mask");
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<cv::String>{"mask_value", "new_sensory", "object_memory"});
    std::cout << "encode_value outputs=" << outputs.size() << std::endl;
    return outputs.size() == 3;
}

static bool ForwardDecode(const std::string& path, int width, int height) {
    const int stride16Width = width / 16;
    const int stride16Height = height / 16;
    const int stride8Width = width / 8;
    const int stride8Height = height / 8;
    const int stride4Width = width / 4;
    const int stride4Height = height / 4;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
    net.setInput(Blob({1, 512, stride8Height, stride8Width}), "f8");
    net.setInput(Blob({1, 256, stride4Height, stride4Width}), "f4");
    net.setInput(Blob({1, 1, 256, stride16Height, stride16Width}), "memory_readout");
    net.setInput(Blob({1, 1, 256, stride16Height, stride16Width}, 0.0f), "sensory");
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<cv::String>{"new_sensory", "logits", "prob"});
    std::cout << "decode outputs=" << outputs.size() << std::endl;
    return outputs.size() == 3;
}

static bool ForwardMemoryReadout(const std::string& path, int width, int height, int memoryFrames) {
    const int stride16Width = width / 16;
    const int stride16Height = height / 16;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
    net.setInput(Blob({1, 64, stride16Height, stride16Width}), "query_key");
    net.setInput(Blob({1, 64, stride16Height, stride16Width}), "query_selection");
    net.setInput(Blob({1, 64, memoryFrames, stride16Height, stride16Width}), "memory_key");
    net.setInput(Blob({1, 1, memoryFrames, stride16Height, stride16Width}), "memory_shrinkage");
    net.setInput(Blob({1, 1, 256, memoryFrames, stride16Height, stride16Width}), "memory_value");
    net.setInput(Blob({1, 1, memoryFrames, stride16Height, stride16Width}), "memory_valid");
    net.setInput(Blob({1, 1, 1, 16, 257}), "object_memory");
    net.setInput(Blob({1, 256, stride16Height, stride16Width}), "pix_feat");
    net.setInput(Blob({1, 1, 256, stride16Height, stride16Width}), "sensory");
    net.setInput(Blob({1, 1, height, width}), "last_mask");
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<cv::String>{"memory_readout"});
    std::cout << "memory_readout outputs=" << outputs.size() << std::endl;
    return outputs.size() == 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-dir> [width] [height] [memory-frames]" << std::endl;
        return 2;
    }
    const std::string dir = argv[1];
    const int width = argc > 2 ? std::stoi(argv[2]) : 256;
    const int height = argc > 3 ? std::stoi(argv[3]) : width;
    const int memoryFrames = argc > 4 ? std::stoi(argv[4]) : 6;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    try {
        const bool key = ForwardEncodeKey(ModelPath(dir, "cutie-encode-key", width, height), width, height);
        const bool value = ForwardEncodeValue(ModelPath(dir, "cutie-encode-value", width, height), width, height);
        const bool readout = ForwardMemoryReadout(MemoryReadoutModelPath(dir, width, height, memoryFrames),
                                                  width,
                                                  height,
                                                  memoryFrames);
        const bool decode = ForwardDecode(ModelPath(dir, "cutie-decode", width, height), width, height);
        if (key && value && readout && decode) {
            std::cout << "Cutie OpenCV DNN load/forward: ok" << std::endl;
            return 0;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    }
    return 1;
}
