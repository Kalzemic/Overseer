
#include "Lookup.cuh"
#include <cuda_fp16.h>

void decode(LookupTable *table, long long *ranks, unsigned long long *addrs, cudaStream_t stream, unsigned long long *anchor);

void encode(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *ip_tensor, long long *page_tensor, cudaStream_t stream);

// void encode(unsigned long long *pages, unsigned long long *ips, half *d_pages, half *d_ips, cudaStream_t stream);

// void decode(half *pred_deltas, unsigned long long *anchor, unsigned long long *preds, cudaStream_t stream);

// void encode(unsigned long long *pages, unsigned long long *ips, half *page_bits, half *ip_bits, cudaStream_t stream);

// void decode(half *pred_logits, unsigned long long *anchor, unsigned long long *preds, cudaStream_t stream);