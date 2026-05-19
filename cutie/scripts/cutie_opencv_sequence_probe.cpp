#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kMemoryFrames = 6;

cv::Mat blob(const std::vector<int>& shape, float value = 0.0f)
{
    cv::Mat out(static_cast<int>(shape.size()), shape.data(), CV_32F);
    out.setTo(value);
    return out;
}

std::string sizeName(int width, int height)
{
    if (width == height)
        return std::to_string(width);
    return std::to_string(width) + "x" + std::to_string(height);
}

std::string modelPath(const std::string& dir, const std::string& stem)
{
    const std::string sim = dir + "/" + stem + "-sim.onnx";
    if (std::filesystem::exists(sim))
        return sim;
    return dir + "/" + stem + ".onnx";
}

cv::Mat imageBlob(const cv::Mat& bgr, int width, int height)
{
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    const int shape[] = {1, 3, height, width};
    cv::Mat out(4, shape, CV_32F);
    float* dst = out.ptr<float>();
    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            dst[(0 * height + y) * width + x] = static_cast<float>(row[x][2]) / 255.0f;
            dst[(1 * height + y) * width + x] = static_cast<float>(row[x][1]) / 255.0f;
            dst[(2 * height + y) * width + x] = static_cast<float>(row[x][0]) / 255.0f;
        }
    }
    return out;
}

cv::Mat seedMaskBlob(const cv::Mat& seed, int width, int height)
{
    cv::Mat gray = seed;
    if (seed.channels() != 1)
        cv::cvtColor(seed, gray, cv::COLOR_BGR2GRAY);
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    const int shape[] = {1, 1, height, width};
    cv::Mat out(4, shape, CV_32F, cv::Scalar(0.0f));
    float* dst = out.ptr<float>();
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = resized.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x)
            dst[y * width + x] = row[x] > 10 ? 1.0f : 0.0f;
    }
    return out;
}

cv::Mat foregroundFromProb(const cv::Mat& prob, int width, int height)
{
    const int shape[] = {1, 1, height, width};
    cv::Mat out(4, shape, CV_32F);
    const float* src = prob.ptr<float>();
    float* dst = out.ptr<float>();
    const int plane = width * height;
    std::memcpy(dst, src + plane, sizeof(float) * plane);
    return out;
}

cv::Mat binaryMask(const cv::Mat& foreground, int width, int height, const cv::Size& outputSize)
{
    cv::Mat mask(height, width, CV_8U, cv::Scalar(0));
    const float* src = foreground.ptr<float>();
    for (int y = 0; y < height; ++y) {
        uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x)
            row[x] = src[y * width + x] >= 0.5f ? 255 : 0;
    }
    cv::Mat resized;
    cv::resize(mask, resized, outputSize, 0, 0, cv::INTER_NEAREST);
    return resized;
}

void copyKeySlot(const cv::Mat& src, cv::Mat& dst, int slot, int channels, int stride16Width, int stride16Height)
{
    const float* in = src.ptr<float>();
    float* out = dst.ptr<float>();
    const int plane = stride16Width * stride16Height;
    for (int c = 0; c < channels; ++c) {
        std::memcpy(out + (c * kMemoryFrames + slot) * plane,
                    in + c * plane,
                    sizeof(float) * plane);
    }
}

void copyValueSlot(const cv::Mat& src, cv::Mat& dst, int slot, int stride16Width, int stride16Height)
{
    const float* in = src.ptr<float>();
    float* out = dst.ptr<float>();
    const int plane = stride16Width * stride16Height;
    for (int c = 0; c < 256; ++c) {
        std::memcpy(out + (c * kMemoryFrames + slot) * plane,
                    in + c * plane,
                    sizeof(float) * plane);
    }
}

class CutieProbe {
public:
    int width = 256;
    int height = 256;
    int stride16Width = 16;
    int stride16Height = 16;
    cv::dnn::Net keyNet;
    cv::dnn::Net valueNet;
    cv::dnn::Net readoutNet;
    cv::dnn::Net decodeNet;
    cv::Mat sensory;
    cv::Mat lastMask;
    cv::Mat objectMemory;
    std::vector<cv::Mat> memoryKeys;
    std::vector<cv::Mat> memoryShrinkage;
    std::vector<cv::Mat> memoryValues;
    int memEvery = 5;
    int maxStoredMemories = kMemoryFrames;
    bool updateObjectMemory = false;
    bool useValidSlots = false;

    void load(const std::string& dir)
    {
        const std::string suffix = sizeName(width, height);
        keyNet = cv::dnn::readNetFromONNX(modelPath(dir, "cutie-encode-key-" + suffix));
        valueNet = cv::dnn::readNetFromONNX(modelPath(dir, "cutie-encode-value-" + suffix));
        if (useValidSlots)
            readoutNet = cv::dnn::readNetFromONNX(dir + "/cutie-memory-readout-floatmask-valid-" + suffix + "-m6-topk30-opencv.onnx");
        else
            readoutNet = cv::dnn::readNetFromONNX(modelPath(dir, "cutie-memory-readout-nomask-valid-" + suffix + "-m6-topk30"));
        decodeNet = cv::dnn::readNetFromONNX(modelPath(dir, "cutie-decode-" + suffix));
        sensory = blob({1, 1, 256, stride16Height, stride16Width});
    }

    void addMemory(const cv::Mat& key, const cv::Mat& shrinkage, const cv::Mat& value)
    {
        if (static_cast<int>(memoryKeys.size()) >= maxStoredMemories)
            return;
        if (memoryKeys.size() == kMemoryFrames) {
            memoryKeys.erase(memoryKeys.begin());
            memoryShrinkage.erase(memoryShrinkage.begin());
            memoryValues.erase(memoryValues.begin());
        }
        memoryKeys.push_back(key.clone());
        memoryShrinkage.push_back(shrinkage.clone());
        memoryValues.push_back(value.clone());
    }

    cv::Mat memoryKeyBlob() const
    {
        cv::Mat out = blob({1, 64, kMemoryFrames, stride16Height, stride16Width});
        for (int slot = 0; slot < kMemoryFrames; ++slot) {
            if (slot < static_cast<int>(memoryKeys.size()))
                copyKeySlot(memoryKeys[slot], out, slot, 64, stride16Width, stride16Height);
            else if (!useValidSlots)
                copyKeySlot(memoryKeys[0], out, slot, 64, stride16Width, stride16Height);
        }
        return out;
    }

    cv::Mat memoryShrinkageBlob() const
    {
        cv::Mat out = blob({1, 1, kMemoryFrames, stride16Height, stride16Width});
        for (int slot = 0; slot < kMemoryFrames; ++slot) {
            if (slot < static_cast<int>(memoryShrinkage.size()))
                copyKeySlot(memoryShrinkage[slot], out, slot, 1, stride16Width, stride16Height);
            else if (!useValidSlots)
                copyKeySlot(memoryShrinkage[0], out, slot, 1, stride16Width, stride16Height);
        }
        return out;
    }

    cv::Mat memoryValueBlob() const
    {
        cv::Mat out = blob({1, 1, 256, kMemoryFrames, stride16Height, stride16Width});
        for (int slot = 0; slot < kMemoryFrames; ++slot) {
            if (slot < static_cast<int>(memoryValues.size()))
                copyValueSlot(memoryValues[slot], out, slot, stride16Width, stride16Height);
            else if (!useValidSlots)
                copyValueSlot(memoryValues[0], out, slot, stride16Width, stride16Height);
        }
        return out;
    }

    cv::Mat memoryValidBlob() const
    {
        cv::Mat out = blob({1, 1, kMemoryFrames, stride16Height, stride16Width});
        float* data = out.ptr<float>();
        const int plane = stride16Width * stride16Height;
        for (int slot = 0; slot < static_cast<int>(memoryValues.size()); ++slot) {
            std::fill(data + slot * plane, data + (slot + 1) * plane, 1.0f);
        }
        return out;
    }

    cv::Mat step(const cv::Mat& frame, const cv::Mat& seed, int frameIndex)
    {
        cv::Mat image = imageBlob(frame, width, height);
        keyNet.setInput(image, "image");
        std::vector<cv::Mat> keyOut;
        keyNet.forward(keyOut, std::vector<cv::String>{"f16", "f8", "f4", "pix_feat", "key", "shrinkage", "selection"});

        cv::Mat foreground;
        if (!seed.empty()) {
            foreground = seedMaskBlob(seed, width, height);
        } else {
            readoutNet.setInput(keyOut[4], "query_key");
            readoutNet.setInput(keyOut[6], "query_selection");
            readoutNet.setInput(memoryKeyBlob(), "memory_key");
            readoutNet.setInput(memoryShrinkageBlob(), "memory_shrinkage");
            readoutNet.setInput(memoryValueBlob(), "memory_value");
            if (useValidSlots)
                readoutNet.setInput(memoryValidBlob(), "memory_valid");
            readoutNet.setInput(objectMemory, "object_memory");
            readoutNet.setInput(keyOut[3], "pix_feat");
            readoutNet.setInput(sensory, "sensory");
            readoutNet.setInput(lastMask, "last_mask");
            std::vector<cv::Mat> readoutOut;
            readoutNet.forward(readoutOut, std::vector<cv::String>{"memory_readout"});

            decodeNet.setInput(keyOut[1], "f8");
            decodeNet.setInput(keyOut[2], "f4");
            decodeNet.setInput(readoutOut[0], "memory_readout");
            decodeNet.setInput(sensory, "sensory");
            std::vector<cv::Mat> decodeOut;
            decodeNet.forward(decodeOut, std::vector<cv::String>{"new_sensory", "logits", "prob"});
            sensory = decodeOut[0].clone();
            foreground = foregroundFromProb(decodeOut[2], width, height);
        }

        if (!seed.empty() || (memEvery > 0 && frameIndex % memEvery == 0)) {
            valueNet.setInput(image, "image");
            valueNet.setInput(keyOut[3], "pix_feat");
            valueNet.setInput(sensory, "sensory");
            valueNet.setInput(foreground, "mask");
            std::vector<cv::Mat> valueOut;
            valueNet.forward(valueOut, std::vector<cv::String>{"mask_value", "new_sensory", "object_memory"});
            sensory = valueOut[1].clone();
            if (objectMemory.empty()) {
                objectMemory = blob({1, 1, 1, 16, 257});
                std::memcpy(objectMemory.ptr<float>(),
                            valueOut[2].ptr<float>(),
                            sizeof(float) * valueOut[2].total());
            } else if (updateObjectMemory) {
                float* dst = objectMemory.ptr<float>();
                const float* src = valueOut[2].ptr<float>();
                for (size_t i = 0; i < valueOut[2].total(); ++i)
                    dst[i] += src[i];
            }
            addMemory(keyOut[4], keyOut[5], valueOut[0]);
        }

        lastMask = foreground.clone();
        cv::Mat mask = binaryMask(foreground, width, height, frame.size());
        std::cout << "frame=" << frameIndex
                  << " seed=" << (!seed.empty())
                  << " mask_pixels=" << cv::countNonZero(mask)
                  << " memory_frames=" << memoryKeys.size()
                  << " mem_every=" << memEvery
                  << std::endl;
        return mask;
    }
};

}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " model_dir video seed_mask output_dir [frames] [mem_every] [update_object_memory] [max_memories] [use_valid_slots] [width] [height]\n";
        return 2;
    }

    const std::string modelDir = argv[1];
    const std::string videoPath = argv[2];
    const std::string seedPath = argv[3];
    const std::string outputDir = argv[4];
    const int frames = argc >= 6 ? std::stoi(argv[5]) : 30;
    const int memEvery = argc >= 7 ? std::stoi(argv[6]) : 5;
    const bool updateObjectMemory = argc >= 8 ? std::stoi(argv[7]) != 0 : false;
    const int maxStoredMemories = argc >= 9 ? std::stoi(argv[8]) : kMemoryFrames;
    const bool useValidSlots = argc >= 10 ? std::stoi(argv[9]) != 0 : false;
    const int width = argc >= 11 ? std::stoi(argv[10]) : 256;
    const int height = argc >= 12 ? std::stoi(argv[11]) : width;

    cv::VideoCapture cap(videoPath);
    cv::Mat seed = cv::imread(seedPath, cv::IMREAD_GRAYSCALE);
    if (!cap.isOpened() || seed.empty())
        return 1;

    std::filesystem::create_directories(outputDir);
    CutieProbe probe;
    probe.width = width;
    probe.height = height;
    probe.stride16Width = width / 16;
    probe.stride16Height = height / 16;
    probe.memEvery = memEvery;
    probe.updateObjectMemory = updateObjectMemory;
    probe.maxStoredMemories = std::max(1, std::min(kMemoryFrames, maxStoredMemories));
    probe.useValidSlots = useValidSlots;
    probe.load(modelDir);

    for (int i = 0; i < frames; ++i) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
            break;
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::Mat mask = probe.step(resized, i == 0 ? seed : cv::Mat(), i);
        cv::imwrite(outputDir + "/cutie_opencv_" + std::to_string(i) + ".png", mask);
    }
    return 0;
}
