#include "encode.cuh"

// __global__ void encode_reg(unsigned long long *pages, unsigned long long *ips, half *d_pages, half *d_ips)
// {
//     if (blockIdx.x >= 2)
//         return;
//     auto data = blockIdx.x % 2 == 0 ? ips : pages;
//     auto target = blockIdx.x % 2 == 0 ? d_ips : d_pages;

//     auto tid = threadIdx.x;
//     if (tid >= SEQ_LENGTH)
//         return;

//     long long a = (long long)data[tid];
//     long long b = (long long)data[tid - 1];
//     float delta = tid == 0 ? 0.0f : (float)(a - b) / (float)PAGE_NORM;
//     target[tid] = __float2half(delta);
// }

// void encode(unsigned long long *pages, unsigned long long *ips, half *d_pages, half *d_ips, cudaStream_t stream)
// {
//     encode_reg<<<2, SEQ_LENGTH, 0, stream>>>(pages, ips, d_pages, d_ips);
// }

// __global__ void decode_reg(half *pred_deltas, unsigned long long *anchor, unsigned long long *preds)
// {
//     int tid = threadIdx.x;
//     if (tid >= SEQ_LENGTH)
//         return;

//     __shared__ long long deltas[SEQ_LENGTH];

//     deltas[tid] = llroundf(__half2float(pred_deltas[tid]) * (float)PAGE_NORM);
//     __syncthreads();

//     if (tid == 0)
//     {
//         long long running = (long long)*anchor;
//         for (int j = 0; j < SEQ_LENGTH; j++)
//         {
//             running += deltas[j];
//             preds[j] = (unsigned long long)running;
//         }
//     }
// }

// void decode(half *pred_deltas, unsigned long long *anchor, unsigned long long *preds, cudaStream_t stream)
// {
//     decode_reg<<<1, SEQ_LENGTH, SEQ_LENGTH * sizeof(long long), stream>>>(pred_deltas, anchor, preds);
// }

// __global__ void encode_bits(unsigned long long *pages, unsigned long long *ips, half *page_bits, half *ip_bits)
// {
//     auto data = blockIdx.x % 2 == 0 ? ips : pages;
//     auto target = blockIdx.x % 2 == 0 ? ip_bits : page_bits;

//     auto tid = threadIdx.x;
//     if (tid >= SEQ_LENGTH)
//         return;

//     unsigned long long delta = (tid == 0) ? 0ULL : ((data[tid] - data[tid - 1]) >> 12);
//     for (int i = 0; i < INPUT_SIZE; i++)
//     {
//         target[tid * INPUT_SIZE + i] = ((delta >> i) & (1ULL));
//     }
// }

// void encode(unsigned long long *pages, unsigned long long *ips, half *page_bits, half *ip_bits, cudaStream_t stream)
// {
//     encode_bits<<<2, SEQ_LENGTH, 0, stream>>>(pages, ips, page_bits, ip_bits);
// }

// __global__ void decode_bits(half *pred_logits, unsigned long long *anchor, unsigned long long *preds)
// {
//     int tid = threadIdx.x;
//     if (tid >= SEQ_LENGTH)
//         return;

//     __shared__ long long deltas[SEQ_LENGTH];

//     unsigned long long raw = 0ULL;
//     for (int i = 0; i < INPUT_SIZE; i++)
//     {
//         unsigned long long bit = (static_cast<float>(pred_logits[tid * INPUT_SIZE + i]) > 0.0f) ? 1ULL : 0ULL;
//         raw |= (bit << i);
//     }

//     // sign extend bit 47 → bits 48..63, then shift back up by 16
//     long long signed_raw = (long long)(raw << 12) >> 12;
//     deltas[tid] = signed_raw << 12;
//     __syncthreads();

//     if (tid == 0)
//     {
//         unsigned long long running = *anchor;
//         for (int j = 0; j < SEQ_LENGTH; j++)
//         {
//             running = (unsigned long long)((long long)running + deltas[j]);
//             preds[j] = running;
//         }
//     }
// }

// void decode(half *pred_logits, unsigned long long *anchor, unsigned long long *preds, cudaStream_t stream)
// {
//     decode_bits<<<1, SEQ_LENGTH, SEQ_LENGTH * sizeof(long long), stream>>>(pred_logits, anchor, preds);
// }

__global__ void decode_seq(LookupTable *table, long long *ranks, unsigned long long *addrs, unsigned long long *anchor)
{
    int tid = threadIdx.x;
    __shared__ long long deltas[SEQ_LENGTH];
    deltas[tid] = (long long)table->addr_lookup(ranks[tid]);
    __syncthreads();
    if (tid == 0)
    {
        for (int i = 0; i < SEQ_LENGTH; i++)
        {
            addrs[i] = i == 0 ? (unsigned long long)((long long)*anchor + deltas[i])
                              : (unsigned long long)((long long)addrs[i - 1] + deltas[i]);
        }
    }
}

void decode(LookupTable *table, long long *ranks, unsigned long long *addrs, cudaStream_t stream, unsigned long long *anchor)
{
    decode_seq<<<1, SEQ_LENGTH, 0, stream>>>(table, ranks, addrs, anchor);
}

__global__ void encode_seq(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *ip_tensor, long long *page_tensor)
{
    int block = blockIdx.x;
    int thread = threadIdx.x;
    auto data = block == 1 ? pages : ips;
    auto table = block == 1 ? page_table : ip_table;
    auto tensor = block == 1 ? page_tensor : ip_tensor;
    __shared__ int ranks[SEQ_LENGTH];
    __shared__ long long deltas[SEQ_LENGTH];
    deltas[thread] = thread == 0 ? 0 : data[thread] - data[thread - 1];
    __syncthreads();
    ranks[thread] = table->lookup(deltas[thread]);
    __syncthreads();
    if (thread == 0)
    {
        for (int i = 0; i < SEQ_LENGTH; ++i)
        {
            if (ranks[i] == -1)
            {
                ranks[i] = table->insert(deltas[i]);
            }
        }
    }
    __syncthreads();
    tensor[thread] = (long long)ranks[thread];
}

void encode(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *ip_tensor, long long *page_tensor, cudaStream_t stream)
{
    encode_seq<<<2, SEQ_LENGTH, 0, stream>>>(page_table, ip_table, pages, ips, ip_tensor, page_tensor);
}
