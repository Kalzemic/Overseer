#include <cuda_runtime.h>
#include "Logger.hpp"
#include "encode.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include "Server.hpp"

void benchmark_inference(
    LookupTable *page_table, LookupTable *ip_table,
    unsigned long long *pages, unsigned long long *ips,
    long long *input_tensor, long long *output_tensor,
    unsigned long long *predicted_pages,
    nvinfer1::IExecutionContext *context,
    cudaStream_t stream);

int main()
{
    // CPU data
    FaultReport buffer;
    EnforcerReport report;
    int state_reset = 0;

    // Server setup
    UDPServer server;
    if (!server.init(PORT))
    {
        std::cout << "could not initialize server" << std::endl;
        return 1;
    }

    // GPU data
    unsigned long long *pages;
    unsigned long long *ips;
    unsigned long long *predicted_pages;
    unsigned long long *anchor; // safe anchor buffer

    half *state;
    half *page_bits;
    half *ip_bits;
    half *pred_bits;
    half *page_bits_trt; // TensorRT scratch-safe copy
    half *ip_bits_trt;   // TensorRT scratch-safe copy

    cudaError_t err;

    // Engine IO
    std::ifstream engine_file("OverNet.trt", std::ios::binary);

    // TensorRT
    struct InferenceEngine
    {
        nvinfer1::IRuntime *runtime;
        nvinfer1::ICudaEngine *engine;
        nvinfer1::IExecutionContext *context;
    } inferenceEngine;

    // GPU Allocation
    err = cudaMalloc(&pages, sizeof(unsigned long long) * SEQ_LENGTH);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate page buffer" << std::endl;
        return 1;
    }

    err = cudaMalloc(&ips, sizeof(unsigned long long) * SEQ_LENGTH);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate ip buffer" << std::endl;
        return 1;
    }

    err = cudaMalloc(&anchor, sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate anchor buffer" << std::endl;
        return 1;
    }

    err = cudaMalloc(&state, 1 * D_STATE * D_MODEL * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate state" << std::endl;
        return 1;
    }
    err = cudaMemset(state, 0, 1 * D_STATE * D_MODEL * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to set state" << std::endl;
        return 1;
    }

    err = cudaMalloc(&page_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate page bits tensor" << std::endl;
        return 1;
    }

    err = cudaMalloc(&ip_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate ip bits tensor" << std::endl;
        return 1;
    }

    err = cudaMalloc(&page_bits_trt, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate page bits trt buffer" << std::endl;
        return 1;
    }

    err = cudaMalloc(&ip_bits_trt, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate ip bits trt buffer" << std::endl;
        return 1;
    }

    err = cudaMalloc(&pred_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate output tensor" << std::endl;
        return 1;
    }

    err = cudaMalloc(&predicted_pages, SEQ_LENGTH * sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate predicted pages buffer" << std::endl;
        return 1;
    }

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to create stream" << std::endl;
        return 1;
    }

    // Print GPU pointer addresses
    std::cout << "pages ptr:         " << (void *)pages << std::endl;
    std::cout << "ips ptr:           " << (void *)ips << std::endl;
    std::cout << "anchor ptr:        " << (void *)anchor << std::endl;
    std::cout << "state ptr:         " << (void *)state << std::endl;
    std::cout << "page_bits ptr:     " << (void *)page_bits << std::endl;
    std::cout << "ip_bits ptr:       " << (void *)ip_bits << std::endl;
    std::cout << "page_bits_trt ptr: " << (void *)page_bits_trt << std::endl;
    std::cout << "ip_bits_trt ptr:   " << (void *)ip_bits_trt << std::endl;
    std::cout << "pred_bits ptr:     " << (void *)pred_bits << std::endl;

    // TRT Setup
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    Logger logger;
    inferenceEngine.runtime = nvinfer1::createInferRuntime(logger);
    inferenceEngine.engine = inferenceEngine.runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    inferenceEngine.context = inferenceEngine.engine->createExecutionContext();

    // Give TRT the scratch-safe copies
    inferenceEngine.context->setTensorAddress("page_bits", page_bits_trt);
    inferenceEngine.context->setTensorAddress("ip_bits", ip_bits_trt);
    inferenceEngine.context->setTensorAddress("state", state);
    inferenceEngine.context->setTensorAddress("pred_bits", pred_bits);
    inferenceEngine.context->setTensorAddress("new_state", state);

    std::cout << "OverNet Running" << std::endl;

    // Warmup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemset(page_bits_trt, 0, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    cudaMemset(ip_bits_trt, 0, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    for (int i = 0; i < 20; i++)
    {
        encode(pages, ips, page_bits, ip_bits, stream);
        cudaMemcpyAsync(page_bits_trt, page_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(ip_bits_trt, ip_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(anchor, pages + SEQ_LENGTH - 1, sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream);
        inferenceEngine.context->enqueueV3(stream);
        decode(pred_bits, anchor, predicted_pages, stream);
        cudaStreamSynchronize(stream);
    }

    // CPU buffers for inspection
    std::vector<half> page_bits_before_cpu(INPUT_SIZE * SEQ_LENGTH);
    std::vector<half> page_bits_after_cpu(INPUT_SIZE * SEQ_LENGTH);
    std::vector<half> pred_bits_cpu(INPUT_SIZE * SEQ_LENGTH);

    while (true)
    {
        if (server.receive(static_cast<void *>(&buffer), sizeof(FaultReport)))
        {
            std::cout << "pid: " << buffer.pid << " input: " << std::endl;
            for (int i = 0; i < SEQ_LENGTH; i++)
                std::cout << "0x" << std::hex << buffer.addrs[i] << "\n";

            cudaStreamSynchronize(stream);
            cudaEventRecord(start, stream);

            err = cudaMemcpy(ips, buffer.ips, sizeof(unsigned long long) * SEQ_LENGTH, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy ips to GPU" << std::endl;
                break;
            }
            err = cudaMemcpy(pages, buffer.addrs, sizeof(unsigned long long) * SEQ_LENGTH, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy pages to GPU" << std::endl;
                break;
            }

            // Encode into clean buffers
            encode(pages, ips, page_bits, ip_bits, stream);
            cudaStreamSynchronize(stream);

            // Copy page_bits BEFORE inference for inspection
            err = cudaMemcpy(page_bits_before_cpu.data(), page_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy page_bits before inference" << std::endl;
                break;
            }

            // Copy anchor to safe buffer
            cudaMemcpyAsync(anchor, pages + SEQ_LENGTH - 1, sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream);

            // Copy to TRT scratch-safe buffers
            cudaMemcpyAsync(page_bits_trt, page_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(ip_bits_trt, ip_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToDevice, stream);

            // Inference
            inferenceEngine.context->enqueueV3(stream);
            decode(pred_bits, anchor, predicted_pages, stream);
            cudaStreamSynchronize(stream);

            // Copy page_bits AFTER inference for inspection
            err = cudaMemcpy(page_bits_after_cpu.data(), page_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy page_bits after inference" << std::endl;
                break;
            }

            // Copy pred_bits
            err = cudaMemcpy(pred_bits_cpu.data(), pred_bits, INPUT_SIZE * SEQ_LENGTH * sizeof(half), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy pred_bits" << std::endl;
                break;
            }

            // Print page_bits BEFORE inference
            std::cout << "page_bits BEFORE inference (first 3 tokens):" << std::endl;
            for (int t = 0; t < 3; t++)
            {
                std::cout << "  token[" << std::dec << t << "]: ";
                for (int i = 0; i < INPUT_SIZE; i++)
                    std::cout << static_cast<float>(page_bits_before_cpu[t * INPUT_SIZE + i]) << " ";
                std::cout << std::endl;
            }

            // Print page_bits AFTER inference
            std::cout << "page_bits AFTER inference (first 3 tokens):" << std::endl;
            for (int t = 0; t < 3; t++)
            {
                std::cout << "  token[" << std::dec << t << "]: ";
                for (int i = 0; i < INPUT_SIZE; i++)
                    std::cout << static_cast<float>(page_bits_after_cpu[t * INPUT_SIZE + i]) << " ";
                std::cout << std::endl;
            }

            // Print pred_bits
            std::cout << "pred_bits logits (first 3 tokens):" << std::endl;
            for (int t = 0; t < 3; t++)
            {
                std::cout << "  token[" << std::dec << t << "]: ";
                for (int i = 0; i < INPUT_SIZE; i++)
                    std::cout << static_cast<float>(pred_bits_cpu[t * INPUT_SIZE + i]) << " ";
                std::cout << std::endl;
            }

            // Copy predicted pages to CPU
            err = cudaMemcpy(report.addrs, predicted_pages, sizeof(unsigned long long) * SEQ_LENGTH, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy pages to CPU" << std::endl;
                break;
            }

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            report.pid = buffer.pid;

            // State reset
            state_reset = (state_reset + 1) % RESET_MAX;
            if (state_reset == 0)
            {
                err = cudaMemset(state, 0, 1 * D_STATE * D_MODEL * sizeof(half));
                if (err != cudaSuccess)
                    std::cerr << "failed to reset state" << std::endl;
            }

            server.send(static_cast<void *>(&report), sizeof(unsigned long long) * SEQ_LENGTH, ENFORCER_PORT);

            std::cout << "pid: " << buffer.pid << " output: " << std::endl;
            for (int i = 0; i < SEQ_LENGTH; i++)
                std::cout << "0x" << std::hex << report.addrs[i] << "\n";

            std::cout << std::endl;
            std::cout << "Elapsed time: " << ms << std::endl;
            std::cout << std::endl;
        }
    }

    // Cleanup
    delete (inferenceEngine.context);
    delete (inferenceEngine.engine);
    delete (inferenceEngine.runtime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(pages);
    cudaFree(ips);
    cudaFree(anchor);
    cudaFree(state);
    cudaFree(page_bits);
    cudaFree(ip_bits);
    cudaFree(page_bits_trt);
    cudaFree(ip_bits_trt);
    cudaFree(pred_bits);
    cudaFree(predicted_pages);
    cudaStreamDestroy(stream);
}