#include <cuda_runtime.h>
#include "Logger.hpp"
#include "encode.cuh"
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
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

    // Server setup
    UDPServer server;
    if (!server.init(PORT))
    {
        std::cout << "could not initialize server" << std::endl;
        return 1;
    }
    // GPU data
    LookupTable dummy;
    LookupTable *page_table;
    LookupTable *ip_table;
    unsigned long long *pages;
    unsigned long long *ips;
    half *state;
    long long *input_tensor;
    long long *output_tensor;
    unsigned long long *predicted_pages;
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
    err = cudaMalloc(&page_table, sizeof(LookupTable));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate pages table" << std::endl;
        return 1;
    }
    err = cudaMemcpy(page_table, &dummy, sizeof(LookupTable), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to copy to pages table" << std::endl;
        return 1;
    }
    err = cudaMalloc(&pages, sizeof(unsigned long long) * SEQ_LENGTH);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate page buffer" << std::endl;
        return 1;
    }
    err = cudaMalloc(&ip_table, sizeof(LookupTable));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate ips table" << std::endl;
        return 1;
    }
    err = cudaMemcpy(ip_table, &dummy, sizeof(dummy), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to copy to ips table" << std::endl;
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

    err = cudaMalloc(&input_tensor, 2 * SEQ_LENGTH * sizeof(long long));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to allocate input tensor" << std::endl;
        return 1;
    }

    err = cudaMalloc(&output_tensor, SEQ_LENGTH * sizeof(long long));
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

    inferenceEngine.context->setTensorAddress("x", input_tensor);
    inferenceEngine.context->setTensorAddress("state", state);
    inferenceEngine.context->setTensorAddress("y", output_tensor);
    inferenceEngine.context->setTensorAddress("new_state", state);

    std::cout << "OverNet Running" << std::endl;
    // Inference Loop
    while (true)
    {
        if (server.receive(static_cast<void *>(&buffer), sizeof(FaultReport)))
        {
            // std::cout << "pid:" << buffer.pid << "input: " << std::endl;
            // for (int i = 0; i < SEQ_LENGTH; i++)
            // {
            //     std::cout << "0x" << std::hex << buffer.addrs[i] << "\n";
            // }

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

            encode(page_table, ip_table, pages, ips, input_tensor, stream);
            inferenceEngine.context->enqueueV3(stream);
            decode(page_table, output_tensor, predicted_pages, stream);
            cudaStreamSynchronize(stream);

            err = cudaMemcpy(report.addrs, predicted_pages, sizeof(unsigned long long) * SEQ_LENGTH, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "failed to copy pages to CPU" << std::endl;
                break;
            }
            report.pid = buffer.pid;

            server.send(static_cast<void *>(&report), sizeof(unsigned long long) * SEQ_LENGTH, ENFORCER_PORT);
            // std::cout << "pid:" << buffer.pid << "output: " << std::endl;
            // for (int i = 0; i < SEQ_LENGTH; i++)
            // {
            //     std::cout << "0x" << std::hex << report.addrs[i] << "\n";
            // }
            // std::cout << std::endl;
        }
    }

    // Cleanup
    delete (inferenceEngine.context);
    delete (inferenceEngine.engine);
    delete (inferenceEngine.runtime);
    cudaFree(page_table);
    cudaFree(ip_table);
    cudaFree(pages);
    cudaFree(ips);
    cudaFree(state);
    cudaFree(input_tensor);
    cudaFree(output_tensor);
    cudaStreamDestroy(stream);
}

void benchmark_inference(
    LookupTable *page_table, LookupTable *ip_table,
    unsigned long long *pages, unsigned long long *ips,
    long long *input_tensor, long long *output_tensor,
    unsigned long long *predicted_pages,
    nvinfer1::IExecutionContext *context,
    cudaStream_t stream)
{
    unsigned long long h_pages[SEQ_LENGTH];
    unsigned long long h_ips[SEQ_LENGTH];

    // warmup with hit data
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        h_pages[i] = i * 4096;
        h_ips[i] = i * 1234;
    }
    cudaMemcpy(pages, h_pages, SEQ_LENGTH * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(ips, h_ips, SEQ_LENGTH * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; i++)
    {
        encode(page_table, ip_table, pages, ips, input_tensor, stream);
        context->enqueueV3(stream);
        decode(page_table, output_tensor, predicted_pages, stream);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // hit benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < 10; i++)
    {
        encode(page_table, ip_table, pages, ips, input_tensor, stream);
        context->enqueueV3(stream);
        decode(page_table, output_tensor, predicted_pages, stream);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "hit avg latency: " << ms / 10.0f << " ms" << std::endl;

    // miss benchmark
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        h_pages[i] = (unsigned long long)9999999 + i;
        h_ips[i] = (unsigned long long)8888888 + i;
    }
    cudaMemcpy(pages, h_pages, SEQ_LENGTH * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(ips, h_ips, SEQ_LENGTH * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    cudaEventRecord(start, stream);
    encode(page_table, ip_table, pages, ips, input_tensor, stream);
    context->enqueueV3(stream);
    decode(page_table, output_tensor, predicted_pages, stream);
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "miss latency: " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
