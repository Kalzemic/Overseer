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

    half *state;
    half *page_bits;
    half *ip_bits;
    half *pred_bits;
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

    inferenceEngine.context->setTensorAddress("page_bits", page_bits);
    inferenceEngine.context->setTensorAddress("ip_bits", ip_bits);
    inferenceEngine.context->setTensorAddress("state", state);
    inferenceEngine.context->setTensorAddress("pred_bits", pred_bits);
    inferenceEngine.context->setTensorAddress("new_state", state);

    std::cout << "OverNet Running" << std::endl;
    // Inference Loop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemset(page_bits, 0.0, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    cudaMemset(ip_bits, 0.0, INPUT_SIZE * SEQ_LENGTH * sizeof(half));
    for (int i = 0; i < 20; i++)
    {
        encode(pages, ips, page_bits, ip_bits, stream);
        inferenceEngine.context->enqueueV3(stream);
        decode(pred_bits, pages + SEQ_LENGTH - 1, predicted_pages, stream);
        cudaStreamSynchronize(stream);
    }

    while (true)
    {

        if (server.receive(static_cast<void *>(&buffer), sizeof(FaultReport)))
        {

            std::cout << "pid: " << buffer.pid << " input: " << std::endl;
            for (int i = 0; i < SEQ_LENGTH; i++)
            {
                std::cout << "0x" << std::hex << buffer.addrs[i] << "\n";
            }
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

            encode(pages, ips, page_bits, ip_bits, stream);
            inferenceEngine.context->enqueueV3(stream);
            decode(pred_bits, pages + SEQ_LENGTH - 1, predicted_pages, stream);
            cudaStreamSynchronize(stream);

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

            // State Reset
            state_reset = (state_reset + 1) % RESET_MAX;
            if (state_reset == 0)
            {
                err = cudaMemset(state, 0, 1 * D_STATE * D_MODEL * sizeof(half));
                if (err != cudaSuccess)
                {
                    std::cerr << "failed to set state" << std::endl;
                }
            }

            server.send(static_cast<void *>(&report), sizeof(unsigned long long) * SEQ_LENGTH, ENFORCER_PORT);
            std::cout << "pid: " << buffer.pid << " output: " << std::endl;
            for (int i = 0; i < SEQ_LENGTH; i++)
            {
                std::cout << "0x" << std::hex << report.addrs[i] << "\n";
            }
            std::cout << std::endl;
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
    cudaFree(state);
    cudaFree(page_bits);
    cudaFree(ip_bits);
    cudaFree(pred_bits);
    cudaFree(predicted_pages);
    cudaStreamDestroy(stream);
}
