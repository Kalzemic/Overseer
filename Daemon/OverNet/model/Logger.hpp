#pragma once
#include <NvInfer.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <string>
#include <fstream>
#include <utility>

class Logger : public nvinfer1::ILogger
{
public:
    Logger() : running_(true)
    {
        logfile_.open("OverNet_logs.txt");
        worker_ = std::thread([this]()
                              { drain(); });
    }

    ~Logger()
    {
        running_ = false;
        worker_.join();
    }

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > min_severity_)
            return;

        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace(severity, msg ? msg : "");
        }
        catch (...)
        {
        }
    }

private:
    void drain()
    {
        while (running_ || !queue_.empty()) // flush remainder on shutdown
        {
            std::pair<Severity, std::string> entry;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (queue_.empty())
                {
                    // release lock before sleeping
                }
                else
                {
                    entry = std::move(queue_.front());
                    queue_.pop();
                    logfile_ << "severity: " << static_cast<int>(entry.first)
                             << ", message: " << entry.second << "\n"; // \n not endl
                }
            }
            if (queue_.empty())
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    std::queue<std::pair<Severity, std::string>> queue_;
    std::mutex mutex_;
    std::atomic<bool> running_;
    std::thread worker_;
    std::ofstream logfile_;
    Severity min_severity_ = Severity::kWARNING;
};