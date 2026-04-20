
#include "Lookup.cuh"

__global__ void decode_batch(LookupTable *table, long long *ranks, unsigned long long *addrs)
{
    int tid = threadIdx.x;
    addrs[tid] = table->addr_lookup(ranks[tid]);
}

void decode(LookupTable *table, long long *ranks, unsigned long long *addrs, cudaStream_t stream)
{
    decode_batch<<<1, SEQ_LENGTH, 0, stream>>>(table, ranks, addrs);
}

__global__ void encode_seq(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *tensor)
{
    int block = blockIdx.x;
    int thread = threadIdx.x;
    auto data = block == 1 ? pages : ips;
    auto table = block == 1 ? page_table : ip_table;
    __shared__ int ranks[SEQ_LENGTH];

    ranks[thread] = table->lookup(data[thread]);
    __syncthreads();
    if (thread == 0)
    {
        for (int i = 0; i < SEQ_LENGTH; ++i)
        {
            if (ranks[i] == -1)
            {
                ranks[i] = table->insert(data[i]);
            }
        }
    }
    __syncthreads();
    tensor[(2 * thread) + block] = (long long)ranks[thread];
}

void encode(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *tensor, cudaStream_t stream)
{
    encode_seq<<<2, SEQ_LENGTH, 0, stream>>>(page_table, ip_table, pages, ips, tensor);
}