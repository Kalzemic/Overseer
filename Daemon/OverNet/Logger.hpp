#pragma once
#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
private:
    const char *severityStr(Severity s)
    {
        switch (s)
        {
        case Severity::kINTERNAL_ERROR:
            return "[FATAL]";
        case Severity::kERROR:
            return "[ERROR]";
        case Severity::kWARNING:
            return "[WARN] ";
        case Severity::kINFO:
            return "[INFO] ";
        case Severity::kVERBOSE:
            return "[VERB] ";
        default:
            return "[?]    ";
        }
    }

public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << severityStr(severity) << " " << msg << std::endl;
    }
};